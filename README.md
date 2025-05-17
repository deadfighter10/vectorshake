# VectorShake: Few-Shot Text Classification via Latent Space Regularization and Dynamic Adaptive Curriculum

Few-shot text classification remains challenging due to the scarcity of labeled data. Standard fine-tuning often struggles, while existing specialized methods can be computationally expensive. We introduce **VectorShake**, a novel training framework designed to enhance model robustness and generalization in low-data regimes.

VectorShake uniquely combines multiple latent-space regularization techniques – including reconstruction, adversarial perturbations, contrastive learning, consistency regularization, and mixup – applied directly to the model's internal representations.

Furthermore, we propose a **Dynamic Adaptive Curriculum (DAC)** that automatically adjusts key regularization hyperparameters during training based on Exponential Moving Averages of performance metrics, allowing the model to adapt its learning strategy.

We evaluate VectorShake on few-shot versions of AG News, GoEmotions, and TREC-6 against standard BERT and SetFit baselines. Experiments show VectorShake achieves strong performance, notably outperforming baselines on AG News, while also demonstrating significant training time efficiency compared to SetFit.

Ablation studies conducted on AG News (K=2) and TREC-6 (K=5) confirm the positive contribution of the individual latent regularization components and reveal a consistent, significant performance improvement from enabling the DAC (e.g., +5.3% accuracy on AG News, +8.0% on TREC-6).

While performance relative to SetFit varies across datasets, the results demonstrate that VectorShake, particularly empowered by the DAC, presents an effective and efficient approach for few-shot text classification.
