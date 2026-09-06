# Instant Checkout Credit Engine - Team Reading Pack

Reviewed: 4 September 2026

## Core reading - everyone

Read these first. They establish the business problem and prevent the team from optimizing an architecture before agreeing on the risk problem.

### 1. The Use of Alternative Data in Credit Risk Assessment: Opportunities, Risks, and Challenges

- Publisher: World Bank Group, 2025
- Why: The best current overview of alternative-data categories, inclusion benefits, business models, privacy and fairness risks, and policy considerations.
- PDF: https://documents1.worldbank.org/curated/en/099031325132018527/pdf/P179614-3e01b947-cbae-41e4-85dd-2905b6187932.pdf

### 2. Credit Scoring Approaches Guidelines

- Publisher: World Bank Group / International Committee on Credit Reporting, 2019
- Why: A practical foundation for score design, data, validation, interpretability, governance, consumer rights, and regulatory oversight.
- PDF: https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf

### 3. How Do Machine Learning and Non-Traditional Data Affect Credit Scoring?

- Publisher: Bank for International Settlements, Working Paper 834, 2019
- Why: Empirical evidence that machine learning plus non-traditional data can outperform traditional approaches, especially under stress, while showing that the advantage changes with borrower credit history.
- Article: https://www.bis.org/publications/working-paper-834-how-do-machine-learning-and-non-traditional-data-affect-credit-scoring-new-evidence-chinese-fintech-firm
- PDF: https://www.bis.org/publ/work834.pdf

### 4. Consumer Use of Buy Now, Pay Later and Other Unsecured Debt

- Publisher: U.S. Consumer Financial Protection Bureau, January 2025
- Why: Direct evidence on simultaneous loans, cross-provider loan stacking, credit-score mix, approval rates, and other unsecured debt held by BNPL users.
- Article: https://www.consumerfinance.gov/data-research/research-reports/consumer-use-of-buy-now-pay-later-and-other-unsecured-debt/
- PDF: https://files.consumerfinance.gov/f/documents/cfpb_BNPL_Report_2025_01.pdf

### 5. The Buy Now, Pay Later Market

- Publisher: U.S. Consumer Financial Protection Bureau, December 2025
- Why: The latest market-level metrics in the pack, covering 2019-2023 originations, users, late fees, and charge-offs.
- Article: https://www.consumerfinance.gov/data-research/research-reports/the-buy-now-pay-later-market/
- PDF: https://files.consumerfinance.gov/f/documents/cfpb_bnpl-market-report_2025-12.pdf

### 6. The Use of Cash-Flow Data in Underwriting Credit: Empirical Research Findings

- Publisher: FinRegLab, 2019
- Why: Independent empirical work comparing cash-flow variables with traditional credit information and examining inclusion and fair-lending questions.
- Article: https://finreglab.org/research/the-use-of-cash-flow-data-in-underwriting-credit-empirical-research-findings/
- PDF: https://finreglab.org/wp-content/uploads/2023/12/FinRegLab_2019-07-25_Research-Report_The-Use-of-Cash-Flow-Data-in-Underwriting-Credit_Empirical-Research-Findings.pdf

## Modeling and multimodal architecture

### 7. XGBoost: A Scalable Tree Boosting System

- Why: The primary paper for the proposed tabular engine; focus on sparsity handling, system design, and how the objective is regularized.
- Article: https://arxiv.org/abs/1603.02754
- PDF: https://arxiv.org/pdf/1603.02754

### 8. DistilBERT: Smaller, Faster, Cheaper and Lighter

- Why: The primary paper for the proposed text encoder. It supports the efficiency argument but does not establish that generic DistilBERT is optimal for noisy transaction descriptions.
- Article: https://arxiv.org/abs/1910.01108
- PDF: https://arxiv.org/pdf/1910.01108

### 9. Revisiting Multimodal Transformers for Tabular Data with Text Fields

- Publisher: ACL Findings, 2024
- Why: The closest architectural reading to the deck. It compares modeling choices for tabular-plus-text classification and proposes a dual-stream model with cross-modal attention and uncertainty estimation.
- Article: https://aclanthology.org/2024.findings-acl.87/
- PDF: https://aclanthology.org/2024.findings-acl.87.pdf

### 10. Specialized Text Classification for Open Banking Transactions

- Why: Useful for data labeling, normalization, domain adaptation, and evaluation of messy bank-transaction descriptions. It is a transaction-classification paper, not a credit-risk validation study.
- Article: https://arxiv.org/abs/2504.12319
- PDF: https://arxiv.org/pdf/2504.12319

### 11. On Calibration of Modern Neural Networks

- Why: A credit engine needs probabilities that correspond to observed default rates. This paper explains why neural classifiers can be miscalibrated and gives a practical post-hoc baseline.
- Article: https://arxiv.org/abs/1706.04599
- PDF: https://arxiv.org/pdf/1706.04599

## Explainability, fairness, and governance

### 12. A Unified Approach to Interpreting Model Predictions

- Publisher: NeurIPS, 2017
- Why: The original SHAP paper. Read it to understand exactly what a SHAP value is - and what it does not establish.
- Article: https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- PDF: https://papers.nips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

### 13. The Fairness of Credit Scoring Models

- Why: Gives a formal testing framework for group fairness in credit scoring and identifies variables contributing to unfairness. It is more directly relevant than generic AI-fairness summaries.
- Article: https://arxiv.org/abs/2205.10200
- PDF: https://arxiv.org/pdf/2205.10200

### 14. Interagency Statement on the Use of Alternative Data in Credit Underwriting

- Publishers: Federal Reserve, CFPB, FDIC, NCUA, and OCC, 2019
- Why: Distinguishes promising cash-flow data from higher-risk alternative data and frames fair-lending, data-quality, privacy, and consumer-protection considerations.
- PDF: https://files.consumerfinance.gov/f/documents/cfpb_interagency-statement_alternative-data.pdf

### 15. CFPB Circular 2023-03: Adverse Action Notification Requirements

- Why: Directly corrects the deck's claim that SHAP feature attributions satisfy regulation. Reasons must be specific, accurate, and describe factors actually used in the decision; generic nearest-match reason codes are insufficient.
- PDF: https://files.consumerfinance.gov/f/documents/cfpb_adverse_action_notice_circular_2023-09.pdf

### 16. Supervisory Guidance on Model Risk Management (SR 26-2)

- Publishers: Federal Reserve, FDIC, and OCC, April 2026
- Why: Current U.S. banking model-risk guidance. It supersedes SR 11-7 and covers governance, model development, independent validation, monitoring, controls, and documentation.
- Article: https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm
- PDF: https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf

### 17. Artificial Intelligence Model Risk Management

- Publisher: Monetary Authority of Singapore, December 2024
- Why: Highly relevant for a Singapore-based team. It covers AI governance, inventory and materiality, development, validation, deployment, monitoring, and change management in financial institutions.
- PDF: https://www.mas.gov.sg/-/media/mas-media-library/publications/monographs-or-information-paper/imd/2024/information-paper-on-ai-risk-management-final.pdf

### 18. Project MindForge AI Risk Management Operationalisation Handbook

- Publisher: Monetary Authority of Singapore and industry consortium, 2026
- Why: The most current Singapore implementation-oriented resource in the pack. Use it to turn governance principles into model inventory, assessment, control, ownership, and monitoring practices.
- PDF: https://www.mas.gov.sg/-/media/mas-media-library/schemes-and-initiatives/ftig/project-mindforge/mindforge-ai-risk-management-operationalisation-handbook.pdf
- Implementation examples: https://www.mas.gov.sg/-/media/mas-media-library/schemes-and-initiatives/ftig/project-mindforge/mindforge-ai-risk-management-implementation-examples.pdf

### 19. NIST AI Risk Management Framework 1.0

- Why: A jurisdiction-neutral structure for governing, mapping, measuring, and managing AI risk. NIST notes that the framework is being revised, so treat this as a stable baseline rather than the final word.
- Article: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10
- PDF: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

## Production and latency

### 20. ONNX Runtime Quantization

- Why: Gives concrete paths to int8 quantization and transformer-specific preprocessing. Benchmark any optimization against calibration and subgroup performance, not latency alone.
- Article: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

### 21. Hugging Face Optimum

- Why: Practical tooling for exporting, optimizing, quantizing, and benchmarking Transformer models on target hardware.
- Article: https://huggingface.co/docs/optimum/index

### 22. FastAPI in Containers and Server Workers

- Why: Use these as implementation references for the API/container layer. Asynchronous HTTP handling does not make CPU-bound model inference asynchronous; benchmark worker count, memory duplication, and tail latency.
- Containers: https://fastapi.tiangolo.com/deployment/docker/
- Workers: https://fastapi.tiangolo.com/deployment/server-workers/
- Async model: https://fastapi.tiangolo.com/async/

## Suggested reading sequence

### Session 1 - Frame the real problem

Read 1, 4, and 5. Agree on the borrower population, BNPL product, target outcome, data actually available at checkout, and consumer harm to avoid.

### Session 2 - Decide whether alternative data adds value

Read 2, 3, and 6. Define the traditional baseline and evidence required to claim improved access or lower losses.

### Session 3 - Design the model experiment

Read 7-11. Write the baseline, ablation, calibration, out-of-time validation, and latency test plan before implementation.

### Session 4 - Design the control system

Read 12-19. Produce a model card, data sheet, fairness test plan, adverse-action reason protocol, validation checklist, monitoring plan, and rollback policy.

### Session 5 - Engineer and benchmark

Read 20-22. Set a measurable SLO and benchmark the full path, including tokenization, preprocessing, both model tracks, fusion, explanation generation, serialization, and network overhead.
