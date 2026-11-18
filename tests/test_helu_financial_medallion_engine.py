"""Tests for Helu.io GmbH Financial Data Platform — Medallion Architecture Engine."""

from decimal import Decimal

from llm_control_plane.helu_financial_medallion_engine import (
    AccountType,
    BronzeIngestionWorker,
    FinancialReconciliationGate,
    GoldFinancialMartBuilder,
    HeluFinancialPipelineOrchestrator,
    MedallionStage,
    SilverDimensionalTransformer,
)


def sample_balanced_voucher_batch():
    return [
        # Voucher 1: Sales revenue on credit (DATEV ERP)
        {
            "voucher_id": "VOUCH-1001",
            "entity_id": "HELU-AT-01",
            "source_erp": "DATEV",
            "posting_date": "2026-11-10",
            "account_code": "1200",  # Accounts Receivable (Asset)
            "account_name": "Accounts Receivable Trade",
            "debit_amount": "5000.00",
            "credit_amount": "0.00",
            "currency": "EUR",
        },
        {
            "voucher_id": "VOUCH-1001",
            "entity_id": "HELU-AT-01",
            "source_erp": "DATEV",
            "posting_date": "2026-11-10",
            "account_code": "8400",  # Sales Revenue (Revenue)
            "account_name": "SaaS Subscription Revenue",
            "debit_amount": "0.00",
            "credit_amount": "5000.00",
            "currency": "EUR",
        },
        # Voucher 2: Cloud infrastructure expense paid via bank (SAP)
        {
            "voucher_id": "VOUCH-1002",
            "entity_id": "HELU-AT-01",
            "source_erp": "SAP",
            "posting_date": "2026-11-12",
            "account_code": "6700",  # Server & Cloud Hosting (Expense)
            "account_name": "AWS Infrastructure Costs",
            "debit_amount": "1200.00",
            "credit_amount": "0.00",
            "currency": "EUR",
        },
        {
            "voucher_id": "VOUCH-1002",
            "entity_id": "HELU-AT-01",
            "source_erp": "SAP",
            "posting_date": "2026-11-12",
            "account_code": "1100",  # Bank Account (Asset)
            "account_name": "Erste Bank Operating Account",
            "debit_amount": "0.00",
            "credit_amount": "1200.00",
            "currency": "EUR",
        },
    ]


def test_bronze_ingestion_valid():
    worker = BronzeIngestionWorker()
    batch = sample_balanced_voucher_batch()
    res = worker.ingest_batch(batch)

    assert res["stage"] == MedallionStage.BRONZE.value
    assert res["total"] == 4
    assert res["ingested"] == 4
    assert res["dlq_errors"] == 0
    assert len(worker.bronze_store) == 4


def test_bronze_ingestion_dlq():
    worker = BronzeIngestionWorker()
    malformed = [
        {"voucher_id": "V-BAD-1", "entity_id": "E1"},  # Missing fields
        {
            "voucher_id": "V-BAD-2",
            "entity_id": "E1",
            "source_erp": "SAP",
            "posting_date": "2026-11-10",
            "account_code": "1000",
            "debit_amount": "NOT_A_NUMBER",
            "credit_amount": "0.00",
            "currency": "EUR",
        },
    ]
    res = worker.ingest_batch(malformed)
    assert res["ingested"] == 0
    assert res["dlq_errors"] == 2
    assert len(worker.dead_letter_queue) == 2


def test_bronze_idempotency():
    worker = BronzeIngestionWorker()
    batch = sample_balanced_voucher_batch()
    worker.ingest_batch(batch)
    assert len(worker.bronze_store) == 4

    # Second ingestion of identical records must not duplicate
    worker.ingest_batch(batch)
    assert len(worker.bronze_store) == 4


def test_silver_account_classification():
    transformer = SilverDimensionalTransformer()
    assert transformer.classify_account("1200", "Trade Receivables") == AccountType.ASSET
    assert transformer.classify_account("1100", "Bank Account") == AccountType.ASSET
    assert transformer.classify_account("3300", "Trade Payables") == AccountType.LIABILITY
    assert transformer.classify_account("2900", "Retained Earnings Equity") == AccountType.EQUITY
    assert transformer.classify_account("8400", "Software License Revenue") == AccountType.REVENUE
    assert transformer.classify_account("6700", "Server Hosting Expense") == AccountType.EXPENSE


def test_silver_currency_translation():
    transformer = SilverDimensionalTransformer({"USD": Decimal("0.90"), "EUR": Decimal("1.0")})
    worker = BronzeIngestionWorker()
    worker.ingest_record(
        {
            "voucher_id": "V-USD-1",
            "entity_id": "HELU-US",
            "source_erp": "QUICKBOOKS",
            "posting_date": "2026-11-15",
            "account_code": "1200",
            "account_name": "US Receivables",
            "debit_amount": "1000.00",
            "credit_amount": "0.00",
            "currency": "USD",
        }
    )
    raw_rec = worker.bronze_store[0]
    fact = transformer.transform_record(raw_rec)

    assert fact.original_currency == "USD"
    assert fact.original_amount == Decimal("1000.00")
    assert fact.amount_eur == Decimal("900.00")  # 1000 * 0.90
    assert fact.fiscal_year == 2026
    assert fact.fiscal_period == 11


def test_financial_reconciliation_gate():
    worker = BronzeIngestionWorker()
    worker.ingest_batch(sample_balanced_voucher_batch())

    valid, info = FinancialReconciliationGate.verify_voucher_double_entry(worker.bronze_store)
    assert valid is True
    assert info["unbalanced_count"] == 0

    # Add unbalanced voucher
    worker.ingest_record(
        {
            "voucher_id": "VOUCH-UNBALANCED",
            "entity_id": "HELU-AT-01",
            "source_erp": "DATEV",
            "posting_date": "2026-11-15",
            "account_code": "1200",
            "account_name": "Receivables",
            "debit_amount": "100.00",
            "credit_amount": "0.00",
            "currency": "EUR",
        }
    )
    valid2, info2 = FinancialReconciliationGate.verify_voucher_double_entry(worker.bronze_store)
    assert valid2 is False
    assert info2["unbalanced_count"] == 1


def test_gold_financial_mart_builder():
    worker = BronzeIngestionWorker()
    worker.ingest_batch(sample_balanced_voucher_batch())

    transformer = SilverDimensionalTransformer()
    transformer.process_bronze_records(worker.bronze_store)

    builder = GoldFinancialMartBuilder()
    stmt = builder.build_statement("HELU-AT-01", 2026, 11, transformer.silver_store)

    assert stmt.total_revenue_eur == Decimal("5000.00")
    assert stmt.total_expenses_eur == Decimal("1200.00")
    assert stmt.net_income_eur == Decimal("3800.00")
    # Assets: Receivables 5000 - Bank credit 1200 = 3800.00
    assert stmt.total_assets_eur == Decimal("3800.00")
    assert stmt.balance_sheet_balanced is True
    assert stmt.discrepancy_eur == Decimal("0.00")


def test_helu_pipeline_orchestrator():
    orchestrator = HeluFinancialPipelineOrchestrator()
    batch = sample_balanced_voucher_batch()
    res = orchestrator.run_pipeline(batch, "HELU-AT-01", 2026, 11)

    assert res["status"] == "COMPLETED"
    assert res["bronze"]["ingested"] == 4
    assert res["reconciliation"]["double_entry_verified"] is True
    assert res["silver"]["processed"] == 4
    assert res["gold_statement"]["net_income_eur"] == Decimal("3800.00")
    assert res["gold_statement"]["balance_sheet_balanced"] is True
