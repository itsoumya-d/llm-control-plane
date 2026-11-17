"""Helu.io GmbH Financial Data Platform — Medallion Architecture Engine.

Enterprise financial data pipeline implementing Bronze, Silver, and Gold data layers
for multi-ERP consolidation (DATEV, SAP, QuickBooks, Xero), automated financial reconciliation,
dimensional modeling (Kimball star schema), and financial statement generation.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import decimal
from decimal import Decimal
import enum
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple


class AccountType(str, enum.Enum):
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"
    EQUITY = "EQUITY"
    REVENUE = "REVENUE"
    EXPENSE = "EXPENSE"


class MedallionStage(str, enum.Enum):
    BRONZE = "BRONZE"
    SILVER = "SILVER"
    GOLD = "GOLD"


@dataclasses.dataclass
class RawVoucherRecord:
    voucher_id: str
    entity_id: str
    source_erp: str  # e.g., "DATEV", "SAP", "QUICKBOOKS", "XERO"
    posting_date: str  # YYYY-MM-DD
    account_code: str
    account_name: str
    debit_amount: Decimal
    credit_amount: Decimal
    currency: str
    description: str
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    ingested_at: str = dataclasses.field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())


@dataclasses.dataclass
class NormalizedTransactionFact:
    transaction_id: str
    voucher_id: str
    entity_id: str
    fiscal_year: int
    fiscal_period: int
    posting_date: str
    account_code: str
    account_type: AccountType
    account_name: str
    original_currency: str
    original_amount: Decimal  # Positive for debit, negative for credit
    exchange_rate_to_eur: Decimal
    amount_eur: Decimal
    is_balanced: bool
    hash_fingerprint: str


@dataclasses.dataclass
class FinancialStatementReport:
    entity_id: str
    fiscal_year: int
    fiscal_period: int
    currency: str
    total_revenue_eur: Decimal
    total_expenses_eur: Decimal
    net_income_eur: Decimal
    total_assets_eur: Decimal
    total_liabilities_eur: Decimal
    total_equity_eur: Decimal
    balance_sheet_balanced: bool
    discrepancy_eur: Decimal
    row_count: int
    generated_at: str


class BronzeIngestionWorker:
    """Bronze Layer: Raw append-only ingestion with schema validation and idempotency."""

    def __init__(self) -> None:
        self.seen_vouchers: set[str] = set()
        self.bronze_store: List[RawVoucherRecord] = []
        self.dead_letter_queue: List[Dict[str, Any]] = []

    def ingest_record(self, raw_data: Dict[str, Any]) -> Tuple[bool, Optional[RawVoucherRecord], Optional[str]]:
        required = ["voucher_id", "entity_id", "source_erp", "posting_date", "account_code", "debit_amount", "credit_amount", "currency"]
        missing = [f for f in required if f not in raw_data]
        if missing:
            err = f"Missing required fields: {missing}"
            self.dead_letter_queue.append({"raw": raw_data, "error": err, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()})
            return False, None, err

        try:
            debit = Decimal(str(raw_data["debit_amount"]))
            credit = Decimal(str(raw_data["credit_amount"]))
        except Exception as e:
            err = f"Invalid numeric amount: {e}"
            self.dead_letter_queue.append({"raw": raw_data, "error": err, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()})
            return False, None, err

        rec = RawVoucherRecord(
            voucher_id=str(raw_data["voucher_id"]),
            entity_id=str(raw_data["entity_id"]),
            source_erp=str(raw_data["source_erp"]),
            posting_date=str(raw_data["posting_date"]),
            account_code=str(raw_data["account_code"]),
            account_name=str(raw_data.get("account_name", "")),
            debit_amount=debit,
            credit_amount=credit,
            currency=str(raw_data["currency"]).upper(),
            description=str(raw_data.get("description", "")),
            metadata=raw_data.get("metadata", {})
        )

        record_key = f"{rec.entity_id}:{rec.voucher_id}:{rec.account_code}:{rec.debit_amount}:{rec.credit_amount}"
        if record_key in self.seen_vouchers:
            return True, rec, "DUPLICATE_IDEMPOTENT_PASS"

        self.seen_vouchers.add(record_key)
        self.bronze_store.append(rec)
        return True, rec, None

    def ingest_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        success_count = 0
        error_count = 0
        for item in batch:
            ok, _, _ = self.ingest_record(item)
            if ok:
                success_count += 1
            else:
                error_count += 1
        return {
            "stage": MedallionStage.BRONZE.value,
            "total": len(batch),
            "ingested": success_count,
            "dlq_errors": error_count,
            "stored_total": len(self.bronze_store)
        }


class SilverDimensionalTransformer:
    """Silver Layer: Cleanses, standardizes charts of accounts, maps currency, and models dimensions."""

    DEFAULT_EXCHANGE_RATES = {
        "EUR": Decimal("1.0"),
        "USD": Decimal("0.92"),
        "GBP": Decimal("1.18"),
        "CHF": Decimal("1.04"),
        "INR": Decimal("0.011")
    }

    def __init__(self, exchange_rates: Optional[Dict[str, Decimal]] = None) -> None:
        self.rates = exchange_rates or self.DEFAULT_EXCHANGE_RATES.copy()
        self.silver_store: List[NormalizedTransactionFact] = []

    def classify_account(self, code: str, name: str) -> AccountType:
        code_str = str(code).strip()
        # European / Austrian standard chart of accounts (SKR03 / SKR04 / Austrian standard KMU)
        first_char = code_str[0] if code_str else "0"
        if first_char in ["0", "1"]:
            return AccountType.ASSET
        elif first_char in ["2", "3"]:
            if "equity" in name.lower() or "capital" in name.lower() or "retained" in name.lower():
                return AccountType.EQUITY
            return AccountType.LIABILITY
        elif first_char in ["4", "5", "8"]:
            return AccountType.REVENUE
        else:
            return AccountType.EXPENSE

    def transform_record(self, raw: RawVoucherRecord) -> NormalizedTransactionFact:
        posting_dt = dt.date.fromisoformat(raw.posting_date)
        fiscal_year = posting_dt.year
        fiscal_period = posting_dt.month

        rate = self.rates.get(raw.currency, Decimal("1.0"))
        net_original = raw.debit_amount - raw.credit_amount
        amount_eur = (net_original * rate).quantize(Decimal("0.01"))

        acct_type = self.classify_account(raw.account_code, raw.account_name)

        # Hash fingerprint for data auditability
        hasher = hashlib.sha256()
        hasher.update(f"{raw.entity_id}|{raw.voucher_id}|{raw.account_code}|{net_original}|{raw.currency}".encode("utf-8"))
        fingerprint = hasher.hexdigest()

        fact = NormalizedTransactionFact(
            transaction_id=f"tx_{fingerprint[:12]}",
            voucher_id=raw.voucher_id,
            entity_id=raw.entity_id,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            posting_date=raw.posting_date,
            account_code=raw.account_code,
            account_type=acct_type,
            account_name=raw.account_name,
            original_currency=raw.currency,
            original_amount=net_original,
            exchange_rate_to_eur=rate,
            amount_eur=amount_eur,
            is_balanced=True,
            hash_fingerprint=fingerprint
        )
        self.silver_store.append(fact)
        return fact

    def process_bronze_records(self, bronze_records: List[RawVoucherRecord]) -> Dict[str, Any]:
        transformed = []
        for b in bronze_records:
            transformed.append(self.transform_record(b))
        return {
            "stage": MedallionStage.SILVER.value,
            "processed": len(transformed),
            "stored_total": len(self.silver_store)
        }


class GoldFinancialMartBuilder:
    """Gold Layer: Builds aggregated reporting marts, P&L, balance sheets, and KPIs."""

    def __init__(self) -> None:
        pass

    def build_statement(
        self,
        entity_id: str,
        fiscal_year: int,
        fiscal_period: int,
        silver_facts: List[NormalizedTransactionFact]
    ) -> FinancialStatementReport:
        filtered = [
            f for f in silver_facts
            if f.entity_id == entity_id and f.fiscal_year == fiscal_year and f.fiscal_period == fiscal_period
        ]

        total_rev = Decimal("0.00")
        total_exp = Decimal("0.00")
        total_assets = Decimal("0.00")
        total_liab = Decimal("0.00")
        total_equity = Decimal("0.00")

        for f in filtered:
            if f.account_type == AccountType.REVENUE:
                # Credit balances in revenue represent positive income
                total_rev += (-f.amount_eur if f.amount_eur < 0 else f.amount_eur)
            elif f.account_type == AccountType.EXPENSE:
                # Debits in expense
                total_exp += (f.amount_eur if f.amount_eur > 0 else -f.amount_eur)
            elif f.account_type == AccountType.ASSET:
                total_assets += f.amount_eur
            elif f.account_type == AccountType.LIABILITY:
                total_liab += (-f.amount_eur if f.amount_eur < 0 else f.amount_eur)
            elif f.account_type == AccountType.EQUITY:
                total_equity += (-f.amount_eur if f.amount_eur < 0 else f.amount_eur)

        net_income = total_rev - total_exp
        # Total equity absorbs current period net income
        adjusted_equity = total_equity + net_income

        # Fundamental accounting equation: Assets = Liabilities + Equity
        discrepancy = (total_assets - (total_liab + adjusted_equity)).quantize(Decimal("0.01"))
        is_balanced = abs(discrepancy) <= Decimal("0.05")

        return FinancialStatementReport(
            entity_id=entity_id,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            currency="EUR",
            total_revenue_eur=total_rev.quantize(Decimal("0.01")),
            total_expenses_eur=total_exp.quantize(Decimal("0.01")),
            net_income_eur=net_income.quantize(Decimal("0.01")),
            total_assets_eur=total_assets.quantize(Decimal("0.01")),
            total_liabilities_eur=total_liab.quantize(Decimal("0.01")),
            total_equity_eur=adjusted_equity.quantize(Decimal("0.01")),
            balance_sheet_balanced=is_balanced,
            discrepancy_eur=discrepancy,
            row_count=len(filtered),
            generated_at=dt.datetime.now(dt.timezone.utc).isoformat()
        )


class FinancialReconciliationGate:
    """Validates double-entry consistency and data integrity before reporting publish."""

    @staticmethod
    def verify_voucher_double_entry(records: List[RawVoucherRecord]) -> Tuple[bool, Dict[str, Any]]:
        vouchers: Dict[str, Dict[str, Decimal]] = {}
        for r in records:
            if r.voucher_id not in vouchers:
                vouchers[r.voucher_id] = {"debit": Decimal("0.0"), "credit": Decimal("0.0")}
            vouchers[r.voucher_id]["debit"] += r.debit_amount
            vouchers[r.voucher_id]["credit"] += r.credit_amount

        unbalanced = []
        for vid, totals in vouchers.items():
            diff = abs(totals["debit"] - totals["credit"])
            if diff > Decimal("0.01"):
                unbalanced.append({"voucher_id": vid, "diff": str(diff)})

        is_valid = len(unbalanced) == 0
        return is_valid, {
            "total_vouchers_checked": len(vouchers),
            "unbalanced_count": len(unbalanced),
            "unbalanced_vouchers": unbalanced
        }


class HeluFinancialPipelineOrchestrator:
    """Full end-to-end Medallion pipeline coordinating Bronze -> Silver -> Gold stages."""

    def __init__(self) -> None:
        self.bronze = BronzeIngestionWorker()
        self.silver = SilverDimensionalTransformer()
        self.gold = GoldFinancialMartBuilder()

    def run_pipeline(
        self,
        batch: List[Dict[str, Any]],
        entity_id: str,
        fiscal_year: int,
        fiscal_period: int
    ) -> Dict[str, Any]:
        # 1. Ingest to Bronze
        bronze_res = self.bronze.ingest_batch(batch)

        # 2. Reconcile Double Entry
        rec_ok, rec_info = FinancialReconciliationGate.verify_voucher_double_entry(self.bronze.bronze_store)

        # 3. Transform to Silver
        silver_res = self.silver.process_bronze_records(self.bronze.bronze_store)

        # 4. Build Gold Financial Statement
        statement = self.gold.build_statement(entity_id, fiscal_year, fiscal_period, self.silver.silver_store)

        return {
            "status": "COMPLETED",
            "entity_id": entity_id,
            "fiscal_period": f"{fiscal_year}-{fiscal_period:02d}",
            "bronze": bronze_res,
            "reconciliation": {"double_entry_verified": rec_ok, "details": rec_info},
            "silver": silver_res,
            "gold_statement": dataclasses.asdict(statement)
        }
