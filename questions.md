## Machine Learning Engineering Q&A

1. Can you design a system end-to-end? From ingestion to serving? What are the bottlenecks?
Yes. An end-to-end ML system includes:

- Data ingestion: Kafka/Flume + batch pipelines (Spark) or stream (Flink).
- Feature store: Feast or custom store over Redis/BigQuery.
- Model training: Orchestrated via Airflow/Kubeflow.
- Model registry: MLflow or Vertex AI Model Registry.
- Serving: Real-time (FastAPI + Triton/TensorRT) or batch outputs into a DB.

### Monitoring & logging: Prometheus + Grafana + OpenTelemetry.

Common bottlenecks:

- Training time due to large data/model size.
- Feature availability mismatches (training-serving skew).
- Latency at inference due to model size or preprocessing.
- Data pipeline lags or schema drift.

2. How would you estimate costs? How to reduce it?
Estimate using:

1. Compute costs: GPU/CPU hours (e.g., GCP/AWS pricing).
2. Storage costs: For raw, intermediate, and model artifacts.
3. Inference costs: Per request (QPS × latency × hardware).
4. Engineering time: DevOps/maintenance overhead.

## Cost reduction:

- Quantize/prune models to lower inference cost.
- Use spot/preemptible instances for training.
- Cache frequent requests.
- Use managed services when cheaper than infra ownership.

3. How would you reduce latency? What is a good tradeoff of latency vs quality?
Latency can be reduced by:

- Model distillation or quantization.
- Using approximate methods (e.g., ANN for vector search).
- Caching embeddings or predictions.
- Parallelizing and batching inference.

Latency vs Quality tradeoff: 
- Accept slight drop in accuracy (e.g., 2–5%) if it improves latency significantly (e.g., 10x), especially in user-facing apps where UX is critical (e.g., search, autocomplete).

4. Do you really need self-hosted LLMs? When is it needed?
Not always. Consider self-hosting if:
You need data privacy/control (e.g., healthcare, finance).
You require model customization (fine-tuning, adapters).
Inference cost is high and you have scale to justify infra.
Otherwise, hosted APIs (OpenAI, Anthropic) are faster to integrate.

5. How would you fine-tune LLMs on user behavior? Which framework? What about model serving?
Use RLHF or SFT on user interactions (e.g., thumbs up/down, click logs).

Frameworks:

- Transformers + PEFT (LoRA/QLoRA) for tuning.
- TRL (Hugging Face) + DeepSpeed for scalable RLHF.

Serving:

- Use vLLM or TGI for optimized multi-tenant LLM inference.
- Deploy via Triton/ONNX + FastAPI, backed by Kubernetes.

6. How would you construct the dataset, what about loss function? What about MLOps?

- Dataset: Combine logs, filter for high-signal examples, label/impute with heuristics or GPT labeling.
- Loss: Cross-entropy for classification, contrastive loss for embeddings, reward models for RLHF.
- MLOps: Use CI/CD with GitHub Actions + MLflow/Kubeflow; track model versions, rollout strategies, monitor drift.

7. Which database would you use and why? Vector DB? SQL? NoSQL?

- Vector DB (Pinecone, Weaviate, Qdrant): For similarity search.
- SQL (Postgres/BigQuery): For analytics, structured queries.
- NoSQL (MongoDB, DynamoDB): For high-volume, unstructured storage.
- Use hybrid: vector DB + metadata in SQL/NoSQL.

8. What metrics would you track? How?

Offline:
- Accuracy/F1/AUC.
- NDCG/MRR (search, ranking).
- Embedding quality (e.g., cosine similarity distribution).

Online:
- Latency, QPS.
- Conversion rate, CTR.
- Feature/data drift (e.g., Kolmogorov–Smirnov test).

Track via Prometheus + Grafana, or ML observability tools like WhyLabs or Fiddler.

9. What about system monitoring? How would you debug failure cases?
Monitor:
- Latency spikes.
- Model inputs/outputs anomalies.
- Resource usage (memory, GPU).
- Retries/errors in serving logs.

Debug with:
- Canary deployments and rollback.
- Logging with trace IDs.
- Shadow traffic to new models.

10. What about feedback loop? How would we track and evaluate?

- Collect user interactions (clicks, dwell time, etc.).
- Log inference inputs + predictions + outcomes.
- Periodically retrain on this labeled data.

Use offline-to-online consistency checks and A/B testing.

11. How would you make the system more deterministic?

- Fix random seeds (in model + data splits).
- Use deterministic libraries (e.g., CuDNN determinism).
- Avoid nondeterministic ops (e.g., multithreaded data loaders).
- Cache preprocessed inputs and model checkpoints.

12. How would you replace embedding models and backfill embeddings without any downtime?

- Dual-write embeddings during rollout period.
- Store versioned embeddings per item (v1, v2).
- Warm-start new embedding index offline → swap via atomic pointer change.
- Use shadow index for testing, monitor drift.

13. What are the fallback mechanisms?

- Use cached responses if model fails.
- Rule-based fallback (regex, heuristics) for critical paths.
- Retry with previous model version on error.
- Graceful degradation (e.g., remove personalization but serve generic content).


