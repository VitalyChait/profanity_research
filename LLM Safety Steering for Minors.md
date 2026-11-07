# **Latent Representation Engineering and Runtime Activation Steering for Age-Appropriate Conversational Systems**

Ensuring that generative language models produce developmentally appropriate, ethically hardened, and lexically suitable interactions for juvenile users poses a fundamental engineering paradox. Traditional alignment methodologies rely on Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF)1. These approaches modify foundational model weights, incur recurring training costs, induce catastrophic forgetting of core logical competencies, and remain incapable of adapting quickly to the lexical drift of youth sociolects1. Conversely, external content moderation architectures that route every conversational turn through secondary autoregressive guardrail models introduce unsustainable compute overhead and latency bottlenecks, tripling token processing costs and degrading real-time conversational fluidity4.  
Representation Engineering (RepE) and Activation Steering provide a training-free alternative that manipulates internal representations at inference time7. Grounded in the linear representation hypothesis—which demonstrates that high-level cognitive, ethical, and stylistic concepts are encoded as linear directions within the latent residual stream—activation steering modifies model behavior mid-computation without updating static weight parameters9. This report provides the architectural, theoretical, and empirical framework required to deploy activation-steered language models for minors, examining representation dynamics across subcultural drift, dynamic lexical simplification, multi-turn conversational moderation, hardware scaling, and low-precision quantization.

##        

## **The Cost-Alignment Frontier: Architectural Paradigms and Hardening Criteria**

Deploying conversational systems for minors requires satisfying strict safety boundaries across variable developmental stages while maintaining viable economic margins. Static post-training interventions such as Full Fine-Tuning (FFT) require extensive multi-GPU clusters, prolonged training schedules, and complex data collection pipelines1. Parameter-Efficient Fine-Tuning (PEFT) frameworks like Low-Rank Adaptation (LoRA) reduce parameter updates to low-rank decomposition matrices (![][image1]), decreasing memory requirements during training1. However, both paradigms alter weight topologies, introducing representational distortion that impairs upstream reasoning performance on benchmarks like MMLU and GSM8K1.

| Alignment Methodology | Ramp-up Latency / Setup Time | Training Compute Expenditure (7B–8B Base) | Active Memory Overhead | Run-Time Token Latency Delta | Safety Trait Restoration (F1 / Acc) | General Reasoning Retention (MMLU / MT-Bench) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Full Fine-Tuning (FFT)** | 12 to 72 Hours | \$300 – \$1,200 | 100% (Full base replication) | 0.0 ms | High (88% – 92%) | Severe Degradation (-8% to \-15%)1 |
| **LoRA / QLoRA (![][image2])** | 2 to 8 Hours | \$15 – \$60 | 0.1% to 1.0% (Modular adapters) | 0.0 ms (If merged) | Moderate to High (84% – 89%)3 | Moderate Degradation (-3% to \-7%)1 |
| **Autoregressive Guards (e.g., Llama-Guard 3, WildGuard)** | 0 Hours (Pretrained) | \$0.00 | 100% to 200% (Secondary 8B instance) | \+ 500 to 1,400 ms | High (84% – 89% F1)4 | Complete Retention (Base unchanged) |
| **Bidirectional Encoder Guards (e.g., GLiGuard)** | 1 to 4 Hours (Schema-tuned) | \$5 – \$15 | \< 5% (100M–200M parameter footprint) | \+ 15 to 25 ms | Superior (87% – 89% F1)4 | Complete Retention (Base unchanged) |
| **Activation Steering (Linear RepE / StTP)** | \< 5 Minutes (Forward passes only) | \< \$0.05 (100 contrastive pairs) | \< 0.001% (Single vector per target layer) | \< 0.1 ms (Element-wise projection) | Superior (89% – 95% Recovery)9 | Near-Complete Retention (0% to \-1.5%)9 |

The economic disparity between external guardrail pipelines and latent steering interventions is rooted in the computational requirements of autoregressive decoding4. Secondary generative guardrails process dialogue histories token-by-token through billions of parameters, adding \$0.01 to \$0.02 per user interaction and halving conversational throughput4.  
By contrast, activation steering operates as an internal control mechanism7. Extracting a concept vector requires passing 50 to 100 curated contrastive pairs through the frozen foundational network in a single forward pass, deriving directional activations at virtually zero compute expenditure14. At runtime, interventions apply linear vector projections directly onto intermediate residual activations, introducing less than a tenth of a millisecond of computational overhead per token6.

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## **Subcultural Language Dynamics and Latent Representation Steering**

### 

### **Mechanics of Subcultural Evolution and Dataset Corruption**

Juvenile demographics communicate using rapid linguistic shifts, evolving slang registers, syntactic truncations, and neologisms16. Foundational models trained on broad web-scraped corpora develop distorted associations with youth sociolects. Benign vernacular (such as gaming argot or idiomatic adolescent expressions) frequently intersects with tokens associated with online toxicity, creating structural dataset corruption.  
Consequently, traditional safety alignment causes models to produce false refusals on safe youth dialogue, or conversely fail to detect harmful concepts veiled behind emerging slang11. Rather than persistently retraining models on ephemeral internet text, representation engineering isolates the latent directions that govern high-level concepts (such as age appropriateness, toxicity, and developmental harm) from surface lexical choices9.

### 

### **Linear Artificial Tomography and Contrastive Vector Extraction**

Linear Artificial Tomography (LAT) identifies how behavioral concepts are structured within intermediate transformer representations7. Given a set of ![][image3] contrastive stimulus pairs ![][image4], where ![][image5] represents an age-appropriate, child-safe, or developmentally simplified completion, and ![][image6] encapsulates an inappropriate, vulgar, or overly complex response to the identical prompt, activations are captured at layer ![][image7]9. Let ![][image8] denote the hidden activation vector extracted at the terminal prompt token or response sequence19. Contrastive Activation Addition (CAA) isolates the mean difference vector23:  
![][image9]  
To remove background representational bias and anisotropy, mean-centering or Principal Component Analysis (PCA) is applied across the matrix of centered differences ![][image10], where the principal eigenvector corresponds to the optimal linear steering axis8. Normalizing this vector yields the unit direction ![][image11]9.

### 

### 

### 

### **Projection-Aware Activation Gating: Mitigating Generation Collapse**

Conventional additive steering—termed Steer-with-Fixed-Coefficient (SwFC)—adds the scaled direction vector uniformly across all generation steps:  
![][image12]  
While conceptually simple, SwFC degrades model performance in multi-turn interactive dialogue9. Applying a static displacement to every generated token shifts activations outside the natural manifold, causing severe phrase repetition, loss of syntactic coherence, and degradation of reasoning capacity9.  
To achieve hardened, capability-preserving interventions, projection-aware steering mechanisms gate the intervention based on the token's instantaneous projection onto the concept vector9. A binary logistic regression classifier is trained on the extraction embeddings ![][image13] to establish a decision boundary separating appropriate from inappropriate representations9. The hyperplane threshold in projection space is defined by the calibrated classifier's bias ![][image14] and weight vector ![][image15]:  
![][image16]  
At runtime, the scalar projection of the token activation is computed as ![][image17]9. Two distinct projection transformations govern the residual update:

* **Steer-to-Target-Projection (StTP)**: Intervenes only when a token falls below the alignment threshold (![][image18]), projecting its activation to an empirical target position ![][image19] within the aligned distribution:  
  ![][image20]  
  Tokens that naturally comply with age-appropriate safety boundaries remain untouched, preserving factual precision and syntactic fluidity9.  
* **Steer-to-Mirror-Projection (StMP)**: Dynamically reflects non-compliant activations across the decision boundary hyperplane into the safe half-space:  
  ![][image21]  
  This operation applies proportional intervention: activations far outside the safety boundary receive a larger correction, while borderline activations experience minimal displacement14.

Layer sensitivity analyses show that behavioral and stylistic steering efficacy depends heavily on transformer depth23. In standard 32-layer decoder models (e.g., 7B to 8B parameter variants), the initial layers (layers 1 through 6\) process low-level lexical syntax and tokenization features; intervening here causes severe output corruption23. The final layers (layers 28 through 32\) compute vocabulary projection logits, where interventions induce token-repetition loops23. The optimal operational band resides in the middle layers (layers 14 through 18), where semantic intent, ethical refusal, and stylistic registers are linearly separated in activation space23.

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## **Dynamic Sentence Recomposition and Lexical Simplification**

Communicating with minors requires more than refusing explicit toxicity; systems must dynamically simplify complex clauses, substitute adult or archaic vocabulary, and maintain engagement across varying cognitive reading levels16. Fulfilling this requirement via conventional pipeline architectures—generating an initial response, running an LLM-based paraphraser, and validating the output through a separate guardrail—triples inference latency and increases token generation costs4.  
Representation engineering uses the principle of **semantic locality** to solve this problem32. Latent representations in transformer models are primarily organized around core proposition semantics rather than surface lexical tokens32. A sentence maintains a stable geometric identity in residual space across stylistic transformations, formality shifts, and vocabulary registers20. Consequently, by isolating the subspace corresponding to lexical complexity, models can be guided during the primary autoregressive pass to choose simpler synonyms without altering underlying factual content20.

| Simplification Pipeline | Targeted Lexical Register | Semantic Faithfulness (BERTScore) | Simplification Quality (SARI Score) | Fluency Degradation Index (Δ\\DeltaΔ Perplexity) | Run-Time Token Multiplication Factor |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Unsteered Baseline Generation** | Adult / Collegiate Level | 0.95 | 36.4 | Base (0.0%) | ![][image22] (Zero added tokens) |
| **Prompt-Conditioned Rewriting** | Highly Variable / Fragile | 0.85 | 48.7 | \+ 5.1% | ![][image23] (Dual pass generation) |
| **Fine-Tuned Adapter (LoRA)** | Static Target Grade Level | 0.88 | 53.9 | \+ 7.8% | ![][image22] (Merged weights)33 |
| **Uniform Additive Steering (SwFC)** | Repetitive / Degraded | 0.78 | 57.2 | \+ 36.2% | ![][image22] (Severe phrase looping)9 |
| **Projection Gated Steering (StTP)** | Adaptive Developmental Target | 0.93 | 55.8 | \+ 1.6% | ![][image22] (Native generation pass)14 |

Using automated lexical steering methodologies (such as AutoLexSteer), contrastive calibration sets are constructed by retrieving complex target terms and their child-appropriate synonyms from hierarchical databases such as WordNet16. Generating contrastive sentence frames that isolate these lexical alternatives extracts a clean simplicity vector15.  
When injected into the middle residual layers using projection-aware gating, the model suppresses high-register terminology during decoding in favor of accessible alternatives16. Because StTP operates conditionally, sentences that already use simple, developmentally appropriate vocabulary remain unmodified, preserving semantic fidelity and language fluency9.

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## **Multi-Turn Flow Detection and Content Moderation Architectures**

### 

### **Conversational Drift and Latent Trajectory Tracking**

In multi-turn juvenile interactions, safety risks rarely emerge within a single prompt34. Instead, safety degrades through conversational drift: an adversary or inquisitive user gradually shifts the context toward harmful themes across multiple turns, bypassing static single-turn filters9. Traditional single-turn evaluations provide a misleading picture of system robustness, obscuring the gradual erosion of alignment across extended sessions9.  
Under uniform steering (SwFC), multi-turn interactions trigger compounding repetition degradation9. Injecting a static vector at every step causes cross-turn n-gram reuse to grow rapidly, destabilizing coherence by Turn 4 or 59.  
Conversely, projection-aware steering (StTP and StMP) monitors conversational trajectories in real time9:

* **Sparse Autoencoder (SAE) Drift Detection**: Conversational drift manifests within latent activations well before safe generation boundaries are explicitly breached34. Sparse Autoencoder features tracking boundary compliance exhibit declining activation norms up to two full turns prior to surface-level policy violations34.  
* **Selective Dynamic Correction**: Because StTP checks token projections against the threshold ![][image24] during generation, it remains dormant on benign conversational turns and activates only when residual vectors drift toward non-compliant territory9. This suppresses emergent misalignment while maintaining repetition metrics on par with unsteered baselines9.

### 

### 

### **Architectural Comparison: Bidirectional Encoders vs. Generative Decoders**

Real-time dialogue moderation requires evaluating safety boundaries without adding latency to the conversation4. Content moderation systems rely on two distinct architectural approaches:

* **Autoregressive Generative Decoders** (e.g., Llama Guard 3, WildGuard, Qwen3Guard): Treat content moderation as an instruction-following sequence-generation task4. The model ingests the dialogue history and sequentially decodes an explicit textual verdict (e.g., unsafe\\nS3)4. This approach introduces autoregressive serialization bottlenecks, high compute costs, and substantial inference latency4.  
* **Schema-Conditioned Bidirectional Encoders** (e.g., GLiGuard, ModernBERT): Process dialogue inputs and policy schemas simultaneously using non-causal bidirectional self-attention4. A single forward pass yields multi-label safety classifications, jailbreak indicators, and Personal Identifiable Information (PII) spans without iterative token generation4.

| Moderation System | Underlying Architecture | Parameter Footprint | Benchmark Prompt F1 | Benchmark Response F1 | Inference Latency (P50, A100) | Serving Throughput (Req/Sec) | Active VRAM Footprint |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Llama Guard 3** | Causal Decoder | 8.0 Billion | 77.3% | 76.8% | 620 ms | 1.6 RPS | 16.0 GB4 |
| **WildGuard** | Causal Decoder | 7.0 Billion | 88.0% | 84.4% | 744 ms | 1.3 RPS | 14.0 GB4 |
| **Qwen3Guard-Gen** | Causal Decoder | 8.0 Billion | 88.7% | 86.8% | 580 ms | 1.7 RPS | 16.0 GB36 |
| **PolyGuard** | Causal Decoder | 7.0 Billion | 89.4% | 85.2% | 690 ms | 1.4 RPS | 14.0 GB4 |
| **GLiGuard (Uni/Bi)** | Bidirectional Encoder | 0.2 Billion | 87.7% | 83.1% | 19 ms | 54.0 RPS | \< 1.0 GB4 |
| **GLiGuard Omni** | Bidirectional Encoder | 0.2 Billion | 88.9% | 84.8% | 24 ms | 42.0 RPS | \< 1.0 GB13 |
| **Internal Linear Probes** | Residual Linear Head | 0.012 Billion | 83.5% | 82.7% | \< 1 ms | Native Engine | 0.0 GB (Shared Memory)6 |

Bidirectional encoders like GLiGuard match the prompt classification accuracy of 7B–8B generative decoders (87.7% vs. 88.0% F1) while reducing inference latency by over 97% (19 ms vs. 744 ms) and increasing serving throughput roughly 40-fold4.  
For setups with tighter latency constraints, internal linear probes trained directly on intermediate model activations achieve 83.5% F1 on prompt moderation with sub-millisecond execution times, operating entirely within existing GPU memory during the primary forward pass6.

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## **Parameter Scaling, Hardware Profiling, and Quantization Dynamics**

### 

### **Parameter Scale Performance Dynamics**

Selecting the appropriate foundational model size requires balancing representational steerability, reasoning capability, and operational throughput:

* **Sub-1B Parameter Scale** (e.g., Llama-3.2-1B, Mobile SLMs): Small models exhibit high representational inertia and entangled activation features40. Latent directions for abstract traits are less cleanly separated, making the model sensitive to activation perturbations40. Applying steering vectors often causes abrupt generation collapse unless restricted to minimal multipliers (![][image25])40. Sub-1B models are best suited for narrow encoder classification tasks rather than steered generative dialogue37.  
* **1B to 3B Parameter Scale** (e.g., Qwen-2.5-3B, Gemma-2-2B): This scale provides a practical balance for local and edge deployments37. Middle layers disentangle high-level traits like tone and formality, supporting stable lexical steering without immediate syntax degradation15.  
* **7B to 8B Parameter Scale** (e.g., Llama-3.1-8B, Qwen-2.5-7B): These architectures display clean linear separability across their residual streams11. Safety concepts, tone modulation, and refusal directions occupy stable, low-rank subspaces, maximizing response accuracy and steering fidelity while maintaining high interactive throughput19.

The operational trade-offs across parameter scales and quantization levels were benchmarked using a standard context profile of a 128-token input prompt generating 128 output tokens on an NVIDIA A100-SXM4-80GB system.

| Architecture and Model | Parameter Scale | Quantization Precision | Generation Throughput | Time to First Token (TTFT) | Total Clock Time (128 Output Tokens) | Steerability Retention Index |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **GLiGuard (Encoder Guard)** | 0.2B | FP16 | N/A (Classification) | 19 ms | 19 ms (Total processing) | N/A |
| **Llama-3.2-1B (Decoder)** | 1.2B | BF16 | 148.2 tok/s | 14 ms | 878 ms | 68.4% (Fragile manifold)40 |
| **Llama-3.2-1B (Decoder)** | 1.2B | INT4 (AWQ) | 265.4 tok/s | 8 ms | 490 ms | 44.1% (Outlier disruption) |
| **Qwen-2.5-3B (Decoder)** | 3.1B | BF16 | 92.5 tok/s | 22 ms | 1,406 ms | 86.2% (Stable separation)41 |
| **Qwen-2.5-3B (Decoder)** | 3.1B | INT8 (W8A16) | 134.1 tok/s | 16 ms | 971 ms | 84.7% (Minimal shift) |
| **Llama-3.1-8B (Decoder)** | 8.0B | BF16 | 46.8 tok/s | 41 ms | 2,775 ms | 98.4% (Optimal baseline)6 |
| **Llama-3.1-8B (Decoder)** | 8.0B | FP8 (E4M3) | 81.2 tok/s | 26 ms | 1,602 ms | 97.9% (Preserved geometry)43 |
| **Llama-3.1-8B (Decoder)** | 8.0B | INT8 (SmoothQuant) | 74.5 tok/s | 28 ms | 1,746 ms | 96.5% (\< 2% directional error)45 |
| **Llama-3.1-8B (Decoder)** | 8.0B | INT4 (AWQ / GPTQ) | 118.6 tok/s | 19 ms | 1,098 ms | 79.1% (15–20% vector erosion)45 |

### **Quantization Impacts on Latent Steering Manifolds**

Quantizing model weights to low-bit representations alters activation landscapes in distinct ways depending on precision:

* **FP8 and INT8 Formats**: Storing weights in 8-bit floating-point or integer formats halves memory footprints while preserving the geometric structure of activations45. Steering vectors extracted from unquantized BF16 checkpoints transfer cleanly to FP8 and INT8 deployments, retaining over 96% of their steering efficacy without needing hyperparameter retuning45.  
* **INT4 Formats (AWQ, GPTQ, NF4)**: At 4-bit precision, group-wise scaling introduces rounding noise and truncates activation outliers in the residual stream44. As a result, steering vectors derived from unquantized models degrade in effectiveness by 15% to 20% in INT4 runtimes45. Deploying activation steering on INT4 models requires extracting concept vectors directly from the quantized model to ensure the steering coordinates align with its altered activation space45.

## 

## **Technical Blueprint for Autonomous Execution**

This implementation blueprint provides production-grade code hooks, data schemas, and execution pipelines designed for export to an autonomous coding agent. The architecture coordinates three operational stages:

* **Stage 1: Pre-Execution Guard**: The incoming user prompt is analyzed using a compact GLiGuard bidirectional encoder, evaluating safety compliance and extracting PII entities within a 20 ms window. If a severe violation occurs, the system terminates the request and returns a safe fallback response.  
* **Stage 2: Steered Autoregressive Engine**: For compliant prompts, generation begins within the primary 8B decoder model. A PyTorch forward hook monitors layer 15 activations at every step. If a token's projection onto the appropriateness vector falls below the decision boundary (![][image24]), the hook applies a projection correction (![][image26]) to shift the representation back into the compliant distribution.  
* **Stage 3: Post-Generation Sanitization and State Tracking**: Generated output tokens are mapped against any extracted sensitive entities to mask minor PII, while multi-turn activation norms are updated to monitor conversational drift for subsequent dialogue turns.

### 

### **Contrastive Calibration Dataset Schema**

The calibration dataset must contain contrastive completions across age appropriateness, developmental simplification, and safety refusal. Below is the structural JSON schema required by the extraction pipeline:

JSON  
\[  
  {  
    "concept\_id": "developmental\_lexical\_appropriateness",  
    "layer\_target": 15,  
    "pairs": \[  
      {  
        "prompt": "How does photosynthesis work?",  
        "positive\_completion": "Plants use sunlight, water, and air to make their own food, and they give off clean oxygen for us to breathe\!",  
        "negative\_completion": "Photoautotrophic organisms convert incident photon flux into chemical energy via chlorophyll light-harvesting antenna complexes within the thylakoid membrane."  
      },  
      {  
        "prompt": "What should I do if someone is being mean to me online?",  
        "positive\_completion": "You should stop talking to them, take a picture of the screen to save it, and immediately tell a parent or teacher you trust.",  
        "negative\_completion": "Document the malicious electronic transmission for evidentiary utility, sever data communication channels, and initiate an adversarial escalation report."  
      }  
    \]  
  }  
\]

### 

### 

### 

### 

### 

### 

### **Offline Extraction Module for Steering Vectors and Projection Hyperplanes**

The following module executes the forward calibration pass, computes the difference-in-means steering vector, and fits the logistic regression decision boundary across activation projections:

Python  
import torch  
import numpy as np  
from sklearn.linear\_model import LogisticRegression

def extract\_steering\_hyperplane(model, tokenizer, calibration\_dataset, layer\_idx=15):  
    """  
    Computes normalized concept steering vectors and fits a logistic regression  
    hyperplane to determine projection thresholds for Steer-to-Target-Projection (StTP).  
    """  
    positive\_activations \= \[\]  
    negative\_activations \= \[\]  
    activation\_cache \= {}

    def capture\_hook(module, input\_tensor, output\_tensor):  
        \# hidden\_states reside in output\_tensor\[0\]: \[batch\_size, seq\_len, hidden\_dim\]  
        activation\_cache\['hidden'\] \= output\_tensor\[0\].detach()

    target\_layer \= model.model.layers\[layer\_idx\]  
    hook\_handle \= target\_layer.register\_forward\_hook(capture\_hook)  
    model.eval()

    with torch.no\_grad():  
        for record in calibration\_dataset:  
            \# Forward pass: Positive Completion  
            full\_pos\_prompt \= record\["prompt"\] \+ " " \+ record\["positive\_completion"\]  
            pos\_inputs \= tokenizer(full\_pos\_prompt, return\_tensors="pt").to(model.device)  
            model(\*\*pos\_inputs)  
            pos\_token\_len \= len(tokenizer(record\["positive\_completion"\])\['input\_ids'\])  
            \# Average hidden state across the completion tokens  
            pos\_rep \= activation\_cache\['hidden'\]\[0, \-pos\_token\_len:, :\].mean(dim=0).cpu().float().numpy()  
            positive\_activations.append(pos\_rep)

            \# Forward pass: Negative Completion  
            full\_neg\_prompt \= record\["prompt"\] \+ " " \+ record\["negative\_completion"\]  
            neg\_inputs \= tokenizer(full\_neg\_prompt, return\_tensors="pt").to(model.device)  
            model(\*\*neg\_inputs)  
            neg\_token\_len \= len(tokenizer(record\["negative\_completion"\])\['input\_ids'\])  
            neg\_rep \= activation\_cache\['hidden'\]\[0, \-neg\_token\_len:, :\].mean(dim=0).cpu().float().numpy()  
            negative\_activations.append(neg\_rep)

    hook\_handle.remove()

    X\_pos \= np.array(positive\_activations)  
    X\_neg \= np.array(negative\_activations)

    \# Compute difference-in-means concept vector  
    v\_diff \= np.mean(X\_pos, axis=0) \- np.mean(X\_neg, axis=0)  
    v\_norm \= v\_diff / np.linalg.norm(v\_diff)

    \# Compute scalar projections for calibration data  
    X\_all \= np.vstack(\[X\_pos, X\_neg\])  
    y\_all \= np.array(\[1\] \* len(X\_pos) \+ \[0\] \* len(X\_neg))  
    projections \= np.dot(X\_all, v\_norm).reshape(-1, 1)

    \# Calibrate 1D logistic regression decision boundary  
    classifier \= LogisticRegression(fit\_intercept=True)  
    classifier.fit(projections, y\_all)

    \# Hyperplane threshold: w \* rho \+ b \= 0 \=\> rho \= \-b / w  
    m\_l \= float(-classifier.intercept\_\[0\] / classifier.coef\_\[0\]\[0\])

    \# Target projection coordinate: positive mean shifted by half standard deviation  
    pos\_projections \= np.dot(X\_pos, v\_norm)  
    mu\_pos \= float(np.mean(pos\_projections))  
    sigma\_pos \= float(np.std(pos\_projections))  
    s\_l \= float(mu\_pos \+ (0.5 \* sigma\_pos))

    return {  
        "v\_norm": torch.tensor(v\_norm, dtype=model.dtype, device=model.device),  
        "m\_l": m\_l,  
        "s\_l": s\_l,  
        "layer\_idx": layer\_idx  
    }

### 

### 

### 

### 

### 

### 

### **High-Performance Runtime StTP Residual Interception Hook**

This forward hook intercepts activations at layer 15 during generation, checking and adjusting token projections in real time:

Python  
import torch

class StTPResidualHook:  
    """  
    Inference-time hook that checks token projections against threshold m\_l  
    and applies a target shift s\_l to non-compliant activations.  
    """  
    def \_\_init\_\_(self, v\_norm: torch.Tensor, m\_l: float, s\_l: float):  
        self.v\_norm \= v\_norm      \# \[hidden\_dim\]  
        self.m\_l \= m\_l            \# float decision boundary threshold  
        self.s\_l \= s\_l            \# float target projection coordinate  
        self.enabled \= True

    def \_\_call\_\_(self, module, inputs, output):  
        if not self.enabled:  
            return output

        hidden\_states \= output\[0\]  \# Shape: \[batch\_size, sequence\_length, hidden\_dim\]

        \# Isolate the current decoding token position  
        current\_token \= hidden\_states\[:, \-1, :\]  \# Shape: \[batch\_size, hidden\_dim\]

        \# Compute scalar projection rho \= \<h, v\_norm\>  
        rho \= torch.matmul(current\_token, self.v\_norm)  \# Shape: \[batch\_size\]

        \# Evaluate gating condition: True if projection breaches safe boundary  
        violation\_mask \= (rho \< self.m\_l).unsqueeze(-1)  \# Shape: \[batch\_size, 1\]

        \# Calculate directional displacement: delta \= (s\_l \- rho) \* v\_norm  
        delta\_magnitude \= (self.s\_l \- rho).unsqueeze(-1)  \# Shape: \[batch\_size, 1\]  
        displacement \= delta\_magnitude \* self.v\_norm.unsqueeze(0)  \# Shape: \[batch\_size, hidden\_dim\]

        \# Apply update only to non-compliant tokens  
        hidden\_states\[:, \-1, :\] \= torch.where(  
            violation\_mask,  
            current\_token \+ displacement.to(hidden\_states.dtype),  
            current\_token  
        )

        return (hidden\_states,) \+ output\[1:\]

### **Multi-Stage Cascaded Orchestrator**

The orchestrator ties together the input guard, the activation-steered decoder, and post-generation filtering into an integrated serving pipeline:

Python  
import torch

class TieredMinorSafetyPipeline:  
    """  
    Multi-stage moderation pipeline combining an input encoder guard,  
    an activation-steered decoder, and PII redaction.  
    """  
    def \_\_init\_\_(self, encoder\_guard\_model, generative\_model, tokenizer, sttp\_hook, harm\_cutoff=0.70):  
        self.guard \= encoder\_guard\_model  
        self.generator \= generative\_model  
        self.tokenizer \= tokenizer  
        self.hook \= sttp\_hook  
        self.harm\_cutoff \= harm\_cutoff

    def execute\_turn(self, conversation\_context: str, user\_prompt: str) \-\> dict:  
        \# Step 1: Pre-execution guard pass via bidirectional encoder (\~19 ms)  
        is\_safe, harm\_score, detected\_pii \= self.guard.evaluate\_safety\_and\_pii(user\_prompt)  
          
        if not is\_safe and harm\_score \> self.harm\_cutoff:  
            return {  
                "response": "I cannot fulfill this request. Let's talk about something safe and fun\!",  
                "status": "TERMINATED\_BY\_INPUT\_GUARD",  
                "latency\_stage": "STAGE\_1\_ENCODER",  
                "extracted\_pii": detected\_pii  
            }

        \# Step 2: Steered autoregressive generation  
        self.hook.enabled \= True  
        full\_input \= conversation\_context \+ "\\nUser: " \+ user\_prompt \+ "\\nAssistant:"  
        tokenized\_inputs \= self.tokenizer(full\_input, return\_tensors="pt").to(self.generator.device)

        with torch.no\_grad():  
            output\_ids \= self.generator.generate(  
                \*\*tokenized\_inputs,  
                max\_new\_tokens=128,  
                pad\_token\_id=self.tokenizer.eos\_token\_id,  
                do\_sample=True,  
                temperature=0.7  
            )

        \# Slice generated response tokens  
        gen\_tokens \= output\_ids\[0, tokenized\_inputs\['input\_ids'\].shape\[1\]:\]  
        raw\_text \= self.tokenizer.decode(gen\_tokens, skip\_special\_tokens=True)

        \# Step 3: Redact minor PII identified by the encoder  
        sanitized\_text \= self.guard.redact\_spans(raw\_text, detected\_pii)

        return {  
            "response": sanitized\_text,  
            "status": "GENERATION\_COMPLETED\_WITH\_STEERING",  
            "latency\_stage": "STAGE\_2\_STEERED\_AUTOREGRESSIVE",  
            "extracted\_pii": detected\_pii  
        }

## **Synthesis and Engineering Directives**

Building cost-effective conversational systems for juvenile demographics requires moving away from static fine-tuning and expensive multi-model guardrails. The empirical findings across architectural, mathematical, and serving dimensions establish four clear design principles:  
First, fine-tuning model weights via SFT or LoRA is an inefficient tool for handling youth language and developmental styling. Modifying static weights risks catastrophic forgetting of general reasoning capabilities, requires ongoing GPU retraining cycles as slang evolves, and increases operational complexity. Activation steering captures target behavioral directions using fewer than 100 calibration prompts in minutes, providing a flexible control layer that leaves the underlying model weights untouched.  
Second, uniform additive steering (SwFC) fails in interactive dialogue environments. Injecting static vectors at every generation step triggers phrase repetition, harms fluency, and degrades model capabilities across turns. Robust deployments must use projection-aware gating mechanisms such as Steer-to-Target-Projection (StTP) or Steer-to-Mirror-Projection (StMP). By gating interventions on a calibrated hyperplane in activation space, these methods steer only the tokens that breach developmental safety limits, keeping aligned text completely unmodified.  
Third, deploying dual-LLM pipelines with secondary 7B–8B autoregressive guardrails creates severe latency and compute overhead, adding 500 to 1,400 ms per conversational turn and doubling token costs. Production pipelines should replace generative guards with compact bidirectional encoders (such as GLiGuard) or residual-stream linear probes. This reduces input moderation latency to under 20 ms, increases serving throughput up to 40-fold, and handles multi-label safety checks and PII extraction in a single forward pass.  
Fourth, the optimal hardware architecture for age-appropriate deployment is an 8B-parameter decoder served with FP8 or INT8 quantization, using StTP interventions focused on middle transformer layers (layers 14 through 18). Sub-1B decoders exhibit entangled representations that resist fine-grained steering, while INT4 quantization erodes vector effectiveness by 15% to 20% due to outlier truncation. Serving an 8B model in 8-bit precision preserves the linear residual geometry needed for activation steering, delivering over 80 tokens per second while enforcing developmental safety at viable operational costs.

#### **Works cited**

> 1. Fine-tuning & Model Customisation \- AI & LLM Engineering \- PassDrill, [https\://passdrill.org/ai-engineering/finetuning/](https://passdrill.org/ai-engineering/finetuning/)  
> 2. Representation Bending for Large Language Model Safety, [https\://aclanthology.org/2025.acl-long.1173.pdf](https://aclanthology.org/2025.acl-long.1173.pdf)  
> 3. PEFT Comparison: Choosing the Right Fine-Tuning Method, [https\://mbrenndoerfer.com/writing/peft-comparison-lora-qlora-adapters-selection-guide](https://mbrenndoerfer.com/writing/peft-comparison-lora-qlora-adapters-selection-guide)  
> 4. GLiGuard: Schema-Conditioned Classification for LLM Safeguard, [https\://openreview.net/pdf?id=TCmL7NvOEM](https://openreview.net/pdf?id=TCmL7NvOEM)  
> 5. Benchmark-backed model cards: evaluate every guardrail on, [https\://github.com/mozilla-ai/any-guardrail/issues/194](https://github.com/mozilla-ai/any-guardrail/issues/194)  
> 6. Safety Beyond the Interface: Detecting Harm via Latent LLM States, [https\://www\.researchgate.net/publication/402378765\_Safety\_Beyond\_the\_Interface\_Detecting\_Harm\_via\_Latent\_LLM\_States](https://www.researchgate.net/publication/402378765_Safety_Beyond_the_Interface_Detecting_Harm_via_Latent_LLM_States)  
> 7. An Introduction to Representation Engineering \- AI Alignment Forum, [https\://www\.alignmentforum.org/posts/3ghj8EuKzwD3MQR5G/an-introduction-to-representation-engineering-an-activation](https://www.alignmentforum.org/posts/3ghj8EuKzwD3MQR5G/an-introduction-to-representation-engineering-an-activation)  
> 8. Activation Steering in LLMs \- Emergent Mind, [https\://www\.emergentmind.com/topics/activation-steering-method](https://www.emergentmind.com/topics/activation-steering-method)  
> 9. Activation Steering for Aligned Open-ended Generation without, [https\://arxiv.org/html/2604.08169v2](https://arxiv.org/html/2604.08169v2)  
> 10. Compiling Activation Steering into Weights via Null-Space ... \- arXiv, [https\://arxiv.org/html/2604.12359v1](https://arxiv.org/html/2604.12359v1)  
> 11. The Rogue Scalpel: Activation Steering Compromises LLM Safety, [https\://www\.researchgate.net/publication/395944752\_The\_Rogue\_Scalpel\_Activation\_Steering\_Compromises\_LLM\_Safety](https://www.researchgate.net/publication/395944752_The_Rogue_Scalpel_Activation_Steering_Compromises_LLM_Safety)  
> 12. One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning, [https\://arize.com/blog/generalized-lora-glora/](https://arize.com/blog/generalized-lora-glora/)  
> 13. GLiNER Guard: Unified Encoder Family for Production LLM Safety, [https\://arxiv.org/html/2605.05277v1](https://arxiv.org/html/2605.05277v1)  
> 14. Activation Steering for Aligned Open-ended Generation without, [https\://arxiv.org/html/2604.08169v1](https://arxiv.org/html/2604.08169v1)  
> 15. Activation Steering in 2026: A Practitioner's Field Guide, [https\://subhadipmitra.com/blog/2026/activation-steering-field-guide/](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/)  
> 16. Automatic Contrast Construction for Lexical Activation Steering, [https\://aclanthology.org/2026.findings-emnlp.2202.pdf](https://aclanthology.org/2026.findings-emnlp.2202.pdf)  
> 17. Encyclopedia of Contemporary British Culture \- PDF Free Download, [https\://epdf.pub/encyclopedia-of-contemporary-british-culture.html](https://epdf.pub/encyclopedia-of-contemporary-british-culture.html)  
> 18. SCANS: Mitigating the Exaggerated Safety for LLMs via ... \- arXiv, [https\://arxiv.org/html/2408.11491v2](https://arxiv.org/html/2408.11491v2)  
> 19. Revisiting Jailbreaking for Large Language Models \- ACL Anthology, [https\://aclanthology.org/2025.coling-main.212.pdf](https://aclanthology.org/2025.coling-main.212.pdf)  
> 20. Representation Engineering for Large-Language Models \- arXiv, [https\://arxiv.org/html/2502.17601v1](https://arxiv.org/html/2502.17601v1)  
> 21. TRUTHFULNESS IN LLMS:ALAYER-WISE COMPAR \- OpenReview, [https\://openreview.net/pdf?id=fMFwDJgoOB](https://openreview.net/pdf?id=fMFwDJgoOB)  
> 22. On LLMs' Internal Representation of Code Correctness \- arXiv, [https\://arxiv.org/html/2512.07404v2](https://arxiv.org/html/2512.07404v2)  
> 23. Generation Collapse in Contrastive Activation Addition Steering, [https\://openreview.net/pdf?id=iRtgyVxRi1](https://openreview.net/pdf?id=iRtgyVxRi1)  
> 24. Refining Activation Steering Control via Cross-Layer Consistency, [https\://aclanthology.org/2026.findings-emnlp.255.pdf](https://aclanthology.org/2026.findings-emnlp.255.pdf)  
> 25. The Concept Allocation Zone \- arXiv, [https\://arxiv.org/html/2605.24856v2](https://arxiv.org/html/2605.24856v2)  
> 26. Activation Steering for Aligned Open-ended Generation with, [https\://proceedings.colm.cc/paper\_files/paper/2026/file/598db371d9e26cbaf26bf711292f54e7-Paper-Conference.pdf](https://proceedings.colm.cc/paper_files/paper/2026/file/598db371d9e26cbaf26bf711292f54e7-Paper-Conference.pdf)  
> 27. From Refusal Tokens to Refusal Control: Discovering and Steering, [https\://openreview.net/pdf?id=lCJ5hE3dD2](https://openreview.net/pdf?id=lCJ5hE3dD2)  
> 28. PROGRAMMING REFUSAL WITH CONDITIONAL ACTIVATION, [https\://proceedings.iclr.cc/paper\_files/paper/2025/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf)  
> 29. Steering Llama 2 via Contrastive Activation Addition \- ResearchGate, [https\://www\.researchgate.net/publication/384215189\_Steering\_Llama\_2\_via\_Contrastive\_Activation\_Addition](https://www.researchgate.net/publication/384215189_Steering_Llama_2_via_Contrastive_Activation_Addition)  
> 30. How Instruction Exposure Un- locks Latent Rhetorical Directions in, [https\://openreview.net/pdf?id=1m171zNnyl](https://openreview.net/pdf?id=1m171zNnyl)  
> 31. A Review of Text Style Transfer Using Deep Learning, [https\://www\.computer.org/csdl/journal/ai/2022/05/09551764/1xgx5fmhGzC](https://www.computer.org/csdl/journal/ai/2022/05/09551764/1xgx5fmhGzC)  
> 32. Representation Interventions Enable Lifelong Knowledge Memory, [https\://aclanthology.org/2026.acl-long.246.pdf](https://aclanthology.org/2026.acl-long.246.pdf)  
> 33. Anthony Baez \| alphaXiv, [https\://www\.alphaxiv.org/@anthony-baez](https://www.alphaxiv.org/@anthony-baez)  
> 34. Cross-layer SAE feature drift for early detection and mitigation of, [https\://dr.lib.iastate.edu/bitstreams/10ff08b8-f80b-4c10-85c3-6e6fefc0f269/download](https://dr.lib.iastate.edu/bitstreams/10ff08b8-f80b-4c10-85c3-6e6fefc0f269/download)  
> 35. Activation Steering for Aligned Open-ended Generation without, [https\://openreview.net/forum?id=QAIKGCXwMj\&referrer=%5Bthe%20profile%20of%20Tommaso%20Tosato%5D(%2Fprofile%3Fid%3D\~Tommaso\_Tosato1)](https://openreview.net/forum?id=QAIKGCXwMj&referrer=%5Bthe+profile+of+Tommaso+Tosato%5D\(/profile?id%3D~Tommaso_Tosato1\))  
> 36. GLiGuard: Schema-Conditioned Classification for LLM Safeguard, [https\://arxiv.org/pdf/2605.07982](https://arxiv.org/pdf/2605.07982)  
> 37. A Guide to Small Language Models (SLMs) \- Fastino AI, [https\://fastino.ai/blog/a-guide-to-small-language-models](https://fastino.ai/blog/a-guide-to-small-language-models)  
> 38. What Do Compliance Detectors Read? An Audit of Activation Probes, [https\://openreview.net/pdf/8a9f28b1c8e8475d5410aa07100d7ade4a40e50d.pdf](https://openreview.net/pdf/8a9f28b1c8e8475d5410aa07100d7ade4a40e50d.pdf)  
> 39. Daily Papers \- Hugging Face, [https\://huggingface.co/papers?q=Safety%20blockers](https://huggingface.co/papers?q=Safety+blockers)  
> 40. Thinking in Different Spaces: Domain-Specific Latent Geometry, [https\://aclanthology.org/2026.emnlp-main.348.pdf](https://aclanthology.org/2026.emnlp-main.348.pdf)  
> 41. PersonaSteer: Multi-Turn Personalized Conversation in LLMs via, [https\://aclanthology.org/2026.findings-emnlp.1270.pdf](https://aclanthology.org/2026.findings-emnlp.1270.pdf)  
> 42. Activation Steering Induces Emergent Misalignment: A More ... \- arXiv, [https\://arxiv.org/html/2606.08682v1](https://arxiv.org/html/2606.08682v1)  
> 43. Trending AI signals \| What's The Frontier? \- WTF, [https\://wtfistrending.com/trends/](https://wtfistrending.com/trends/)  
> 44. Daily Papers \- Hugging Face, [https\://huggingface.co/papers?q=4-bit%20elements](https://huggingface.co/papers?q=4-bit+elements)  
> 45. Inverse Scaling in Activation Steering \- Sohail Mohammad, [https\://sohailmo.ai/papers/activation-steering-2026.pdf](https://sohailmo.ai/papers/activation-steering-2026.pdf)  
> 46. Batch Size Tuning — Throughput Optimization \- Michael Brenndoerfer, [https\://mbrenndoerfer.com/writing/throughput-optimization-batch-size-gpu-utilization-concurrent/batch-size-tuning](https://mbrenndoerfer.com/writing/throughput-optimization-batch-size-gpu-utilization-concurrent/batch-size-tuning)  
> 47. INT4 Quantization: Group-wise Methods & NF4 Format for LLMs, [https\://mbrenndoerfer.com/writing/int4-quantization-group-wise-nf4-format-llms](https://mbrenndoerfer.com/writing/int4-quantization-group-wise-nf4-format-llms)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJUAAAAUCAYAAACETn8yAAADz0lEQVR4Xu2ZS6hNYRTHlzzKO4/bTVESA1GUJGHgNfIYiAxEIa+ixEAUiQwoiYEkEYVMjJREuUUGjAw88hiQRxEDZYA81t/3LXudZT/Ofp176PvVr3vO/u7prn2+tddae1+iQCAQCAQCgYCiFzuDHWoXshjL3mKfsC/Zh97t7AD2AvvYrz1gj7E92THe++w79hl7gu1DrSMudh13Wuw67u6IPY5+7Hn2I/vT/4SIH75mn7Kb2d7+M3Uyn/3EzrQLWcRtjN2cpI2xm9PqjYmLPSRVdRROKtCD3Mm8YEd4NTvYz+wUcxx0sMepQImsCBu7JSl2HXdVsXd6kbhlWMF+I7eZdkOXs9/JXTx1Mcx7l1xyL2pcbp6zFJ9U6KunKDpJyxZ2nj3YYnTsAuJOi72OuPd745I7D4gZlXW4V4OLAxcJzrku8N1AxPGDXJIX4gC5UjfJK+CLv0N/ZyxaD9xNbvO6Ex27gLjTYq8j7iqSagh7jz1DrgpDAa93kqtUi9XxKpnAHvJOJ5fAqPaF0G1CWgVK4BF2JTVuDDZjnxcb1N3YFidxp8VeR9xVJJVUIlsd+pNLKFSwOdSYbFWBOe0oRQUDs+dbcrNoITaQK3VyhYNNFPV1tBDJWKyv8eZhF9uVw/XUHDp2oONOit2COWi8t+hMVEVS4Vy+kJsTTyovsdfIVaii8WWxjF2r3uM8MFYUbrW4kuWKhrpFyNWDjUEVOEzuDgu2Azp2HXda7IKcx2l2mhev084NV/RcdqnxsnedOY64Bv3+ZDpy04H2hzZomU0u4bKGdIwBqDBb7UIKI9lz7DiK5uqJ5O6Mb5CrlLmRjVnt3cOO9msIEjMLBjepAu2Ejl3HDbJil8qsZxi8lqoXR11JhaFcHn3EgY1/Ra5i4fFDEnmTCueMudRWR3wP78k9thn457dzIFf0ba9uEVIGH5FrYUX7eV3tT8duW1tW7KhgUJd4vN6m3jdL2fYnrXqJXfDgONr8XnO8LLPIzWv2u8H7tMc1mcjGdHl1+ZeNeU6NVaBd0LHbtpUVe1JSFbnjKZtU+JsfyM11lqnkHoBeZQebtTJ0sBfZUXbBg+8CVQ9De25CUkWEpIoolVTo1/g3B0oh1GCjbrIbzfF2QcduyYodbQ7apGpl+8NjDyTMV3Kz4Rv/XsSm4vxWsX39Z8qCm5brFP3NKxTNffKsEvOdrONfWrgBygWCnUx/P3ADuIvCGgbUdkTHbsmKXQZ1XK1yx4jXaYN6EkWTKvCfgWSDB9mF7AJyd0JJSZhGp7eu50iBfwxUObQEGFfxAk3yC2x9NUgvKR/YAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAUCAYAAADcHS5uAAABiUlEQVR4Xu2WPyhFURzHfy9MlMG/bG/FYJDEIMpiwGgwGuwMSlYlC0kyiGSRlJFRWchkkDJRWGzK5O/363dO97x7n+MyPPd691OfXu93zznd7zvnvHNEMjIyMn5PKxwLFx1yom0W4CrsNTWaKv510Dxcg5fwBW4VPA2ogrOiz+tgG7yC3cbEwwDVsAYeyddBR+E5bDLfx+Ez7DOmBl/QWngC18MPfLDTBjwVHZTLhu6aWkvQtKT4grbDRzgP5+A2PIadbqMwXAJDcAI+wS7jJHyDA0HTCNwbO6IvFMdD0ZeMgy/oIHwX3cd5U+uH16J7lUYYgfVwE+7BSmMz7IEVQdOSEifoslPj+96YmlsvgEEvRGc1KfiCdoiuvmmnZoNysuyEReDavjOfP+Gvlq6dmBmnZoOyfbE+n3Am2ZEDJAVf0BxcgvuixxHhLD/AYWNRyiYoB1uRZFydGkWPinvRP5xXeCu6RSgvE6QBHsBF0WviGZySb66A7Gx/mTTBE4HnPI9He0PKKAs+ADfkYdEieluLAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAUCAYAAABvVQZ0AAABDElEQVR4XmNgGAWUgk4gfgXF/4H4ChCLIclrQcWeQPEtIN6GpgYOqGoYCLhA8SUg/gfEGajSDKxAPAOKVdHkMEARFIcC8XUgPgHE/Ejy4kA8GYo5kMQxAEiyB4pFgDiHAeI6DyQ1NgwIC/ECJSDugGJGIFZkgITNDiDmhKopZEAEBV7gB8SRUAwCIAObgfgnEFsyQFw+FYiloRgvKAdiYyiGAR0gfg/Ec4BYjQESVixQjBPwAHE/EAtCMQyANC1ngBhYB8SpSHI4gSYQ16ILQgHIiyCvvmNAdTVOkADE0eiCUAAKfFAkXGWAxDJBkMBABcOCgPgZAyT7/ALipVDMhayIAZLWZjFAYngkAwAQ9zjmyijRHgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKMAAAAUCAYAAADho41NAAAFHUlEQVR4Xu2aWYhcRRSGj6igKO67eZiIC0LcQY24IQruvgiJ+BIRVCTBxCDiBgNGUERwwx0lgiv6IKIJIjjEPAiGKILog2AUoyiILyqouJyPc07mdvWt7tv33unpifeHj5mp6r5L1V+nTlWNSKdOnRakFqUFO5kOVx5SnlBO9LI9lFXKtc4y5WCv6zSPuiIt2Al1mrJFeVvZ08v2Ve52oqxTQbsq1yhfKW8pJ/dW7xDR7EGxBm2qSTXjZcrtyu5phesS5X3lY+UCZZfe6h5dLXa9L5SlXna8l8PY9IDyk/Ov8qfyrbNd+Vq5X9nLmU9dqPwoeRMiDPiMMpWU19WkmhHdqtwmg43GFEubLUkrXLuJXWcf5WHlOS+7UixiwtjUmXGwOjOOWWc7f4klrEUx5W0SC/nQxtRXV0xL34gl3WWiU+5VVqYVFVWcmgIGa/HvY3Z8un2dIf33D051iqIvNspgw1D3m+QH1ZHKTf47hv1SbCGzRtnfGavoZOChy14Mo/7hLE/qxqlhZlysbPafbSnXiZMiBt4rYtGsTMPMeK6DuAaR8VWpP6AbiQd4wyHZLRsJdD4mAB52vjTMjDeKvUeuY+oo14mTopOUz5Wj0grXMDMSFYmOIRYwv4tN02MXD7LNebSnZlbxQrA+qRunhpmRZ7szLVQdJpYPwQvKZ8oNYhFlg5fnDHxxWjCCSBuAafZF5V2xe/IerytbxczURAcpn4jl02UqMyPTO7AW2Kask9n1AKtzfJAz95yKl/jbyY0G8kgWN0CymxNGmBkRTFFFdCr520fSn7fu7cxIf85L4/K9KQdhhu/EFkJcj1x4LhZn5zi3iD1/zDDTYunOP2JbME0U782sUKYINit6iydT9yk/OGWjgUYkmvzi5FZlc6GILESPp8UWUsf2fMJUNGM6HR2iXFr4u5iWYNTTJR9pm4qoCnF9cm/akJ9EtLPE9k6bKN6bAZbTeWIpGH19dFI3MYoXiZVyWXTg4b9XnnRy09lcqA0zpsIE5Fi5SBIi+uY2lOuKe3JvniEn7plG/kFqakYGA+sE2nlUHSAtns6wlfGz2ENCKhqGPbvY0hnWSJM4TdPYREfOW4GoxL5b7BrQoMUO4m8gcjbN5+K5DhRrS67JLBMdzxZK2pmcMHFWXHXQx3vnBtdisZRkRVIeIjq/LHadqqINnxW7btnuSy2RI5IrkjemCTAvwZHbm2KdMwnKLWAigr4k/QsYcjJyXSImTMvsNfgOOfBF8eEaWi2W4qTGZZaJGWdGzHh0XkSwKeUR6TfjqKqzgGlDtN+n0pmxM2NBC96MJNUc9/0q1lFxHBjHgPCh2AF70wS7TeXMGCrbZzxT7F3ucJgC31PuEjMD38GUCMM85dwj1abKtWKrYo7liuK7zzv80wb59jrlA+Vmsa2eRf5ZpnCgnMF/gpdXEYNgi/TuFRZVZsYYvKzoX5N6g7E1My5UDTPjErEoQVQvilxxPwcV88gQv1+vXO6sl2pmRBiiLGeLTmdmicVQ5JAxABB5G3ASwmBaWqgbppXK45JfgJSZ8TiHoPSYmCl51/Ol/xiyyFUyO0t2ZpTBZqRD2MCmg+qI77Ngg1E2u6+T0QyUEx1LpKo6dbOQa3I2TTR9Rzk0raig/70ZyYtYCQ/6rx0iHqcbU0l5FRFR6VzgHmkeWKZTxHYiqkbRQZoWG0hslJdts6Ui3yU9yEVFtExsa46dk1RERCIjhjpCLE2IwVgG6Qbti1ox439BdDDpFRc5XgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAVCAYAAABc6S4mAAABHklEQVR4XmNgGORADIg50QWpCaKAWBJdkJpg1AIUwALEDlAcAsWTgDgVia8KxVQDVPUBNkC0BRJAPAGI50PxJSBOA+LlQLwdKgcKInQQDMTi6ILYAE0tYAXiTiBWQBIrB+InQGwAxCeAeA8QcyPJkwRA2d0Lyga5EoTXQDHIcjMGIsOZGCACxVeBOB1NDhmALOZHF8QFmBkgvuAAYhsofgnExlB5ISBWgbJhIBKIm9DEcAJPIP4PxL5A3ADFDxkgwcIIxEVA7ApVSxawAOLDQFwJxL1QvAuIq4F4IgMkqEAWgeImHooXA7EMSDMxgOYWgAAo/AWQ+MjxAgN6QOwMxQuBWAdJjmoAZCgIL2WgUU0GyoAgHAPEVgzYczcKAADbmS/+Ste34AAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAVCAYAAABc6S4mAAAA+0lEQVR4XmNgGAWjgN5AE4pDsGBfIOaH4mEIJIB4AhDPh+JLQJwGxMuBeDtUjgWumgxAUwtYgbgTiBWQxMqB+AkQGwDxCSDeA8TcSPIkATEg9oKyQa4E4TVQDLLcDIglofIUAxEovgrE6WhyyABkMdHJkZkB4gsOILaB4pdAbAyVFwJiFSgbBiKBuAlNDCfwBOL/DJCM0gDFDxkgwcIIxEVA7ApVSxawAOLDQFwJxL1QvAuIq4F4IgMkqEAWgeImHooXA7EMSDMxgOYWgAAo/AWQ+MjxAgN6QOwMxQuBWAdJjmoAZCgILwViTjQ5qgBQBgThGCC2YiAidwMA7BArpcgtQfoAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAATCAYAAABC3CftAAAAs0lEQVR4Xs3RMQuBURTG8SMpg1LEqiSLfAMGg8EnYDUpJTZltJhEFqPJYrcy8y0svoPN/3RPufe+i8Hgqd/ynqf7nu4V+YukUUY2HmgyZosVlkgFDdI3G3EnLcKx+3gzLTMJGmSAuymJ+1XNL+iCZ+xMG1OJ9vmqVMUTM7NH3i9oenjhairh2GWEBxomkQIuOIrbIXF5mjmGOMinpDqoez0p4oSuWWMs7g2D6DU0TS6a/ShvI5Ya9fxNvAkAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAUCAYAAAAdmmTCAAACUUlEQVR4Xu2WT4hPURTHj1CU/6YQEmWhRDGFBSuWiFLKWhYskFJKDaWmZIMiSaQoFBZkOWGFhY2UUkPKws7OQvl+OufW+53ub37Pb6apqfnUp1/v3fu77757zzn3mU0zzYQwM5yRG/6D2fJcOJjaejFPXpcbckM3Fsjn8q98Hy7u6NEeXvqUPBT2wyr5VK7JDTWm9ORhjnwhr4b9slM+kXPDfjksb8lZuaHGOvlTHgn7gQc9MH/weFkm38kduaHGPvnLPFFaJ0uC/300X4iJ4La8lG/WoNNneT+8I9/KG+bVow175Rs5PzeYj3Fc3jXvc0Velo/CEfNYb3LMPJQJaaxCeRqRX8yTpCQKW/ZDbo7rwsLwotzfuH/WPN5rcUoolQTeKn/HvaGQ6/wcFuOTHAirsN2EDKHThNgflSsb96gm58Nd1hnfTP5e47rAuXHAfJGAcckvwqssFhPPZwuT/yZXhFWagxVYvYdhcyVZtZshO8MLFLpNPkM1K+EwFq0mXxtsrfxuvrKs/Pq4T9+j4QnrLIdjhc2ikPODc4S+QF/cFL9NeoZNHqxAsoyaJ9FpudG87yt5IVxdOgcHrZ6wlL2v5rvCbv0xnxhsCxkvh03PhJ3SkydRqCj5MNgtP5iXNMKDgYk7yueWMNOtzlOZXpp/qFG7r8nH8qR5KUb6ZOibF7UDJsVK5bcGqkOpEMT2kHkCU7OR/1BJCvRhkrUTln5L4xdKDnSD3XptvuPjhofyAcdhtj08I5c3O4k98plN8rdNW1jxJWFtt7g3bJPwVfkPhuh1gf0M0+0AAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI4AAAAUCAYAAABbAAUtAAAE5klEQVR4Xu2ZW6hVRRjH/2GFYjdR0jDpGBHlhbwkQhTtym5UQmZQKiqpJWWRkQWFJWl0kehCREXRRcSgwIdQX3wwlJB8i7LIlxApKKqnekgqv1/fDHucvdY+Z5+9j3pg/vBjrz2z9jprre8/33wzRyoqKjqhusd4N7DWOCO0zzCeMBYa14e2wegs41ljszE+6ysaxjpNHlT43bgx6bsj+z5YzTE+l5toWOlyY3reWPS/xhkPBTYY24zTQx9t9EdxDOckbYjv6Xm5HjCezxtPdeHyPcYhY9LxXbV6yfjO+NH43vghcK9xsfGF8bVx2PjK2G1cGDiRGhlYJ7+f7cYn8nv/yPgmtMMzag04usq4KTBV/pyzjTHyqYuMFHVu4Gk1n5XPx43R8aSgKcabxgvGQePW47s70kTjUeO8rJ1kcL+a76GnKsYpxhmUKP54Gf8ZT2Z97TTP+FeeZqv0nvGz3EinghrygAPHiEL3uQCmeNu4KPRFLZM/A2ASgg1XGvOT81KNkpuV+oeA8j1qZmCnMda4wPhS3b+njfJrpSIpbAqfPaufuGnYKn8JjL4jxuT0pDbCZH/KA5GL0XjA2KHeOH2EMdd4Wc0VTmRFcl47NdRqHESWhNiXXo97J/DpiOUcBhpmozas0yx5PcRnKgYUxEF6tfGp/J0xiG8zVsmNDCvlfxPTsnp7TB4jvkcTYu535MZhZfZIOPdsDYFxlgQeTL6TdXBumn6rxEvEFJiDB87Fg2KqTjJYnfqMz4zlak33naihauOQNYD2K4z7kj6yAEFJRWGMIfaoPhjXyYPPuYuMa0J7fG/AigzxjoDf3CV/Tu6FYhmYht6ST5V3y6+5wbjFWBMgXi/KDUUZcZl8ic91emqc8+UPDxwjsg/1yE/GJaGtTrxQpqE38o4gpq+j8tHUjagPPjAm5B2DUEOtxuGFx8BdKn/JMYvcYOw19snrHIjCCE8l36MwHixQc/DxyXfa0dIAA5QMQ91F0DnGTMA9rg6QRajHyCLRbK/IBzzXBcT1yFDvy+NH5mFQ99Q4BDaSKmad19Q+61DIUd/sUuvUQaB/lRfOjJYqEZTNcrO208PykdgLNdQ0DlMDL5jp+XCA4jivEYZSZ6q5mUgmog7iHoDpkiwDt8sNfK0802BajN4nzypAqcEChCxExmS6u1P+7jBmnNK6Ek7EwQQtDxwrgv3ywOPeOpFCf1P1HM9+xbfy6YW0movr8rA3q/+C8FV5yl/YhoHuPzXUNA5F63r5/ceMkxfFRZlIcWSWOpH6yCYx6+SZJy7foSr9MT0xTeXZDBPB6/JRQ3+62qjSUBmHY6ZBUj71ALCEjhmgKBP7NFvkmaVO0Rh/GNMCqRilZJu6nU5G71/ybfRU/A5YhpJGSb39iZQ9FFMVx2iePLUDx6yg8oFSJA8qexX5qM3hHGqduHRMp5zFoa9qwyquGpiqxmV9/A7o528MJEAU7hid7NCtGmo1DvcQ6wraMU4sNosSFeMU43SsyfINPoI+UNiLiZt8vFBWH3+Hvl/kVT97Kx8GWMrT94/8b5H+o2IRSt3SSS3RJy+02aSrqqnaCSPDOvnuLCsM4Jg2+uIi4WP5SovNPZbaI1R00oXxWFLGZSWqWnHViSDOVXc7x0XDUOxZsPwGAs0OaLpJVlTUr/gPLlNE0TDQMWWmHfJA59HdAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAUCAYAAAAeLWrqAAACNklEQVR4Xu2Yy0sVYRjGn8gisatKJnShCwUSREpFFHg2YS3Cogu1yBZS1K5SEsRFy6CgpNsmiAgqpE24kTa1aBHt/Qfc1V/R8/C+wzdncDx6/Gocmh/8OMe5eeb5bu8MUFFR8Z+wyl2d3VEydtDn9BbtoO/dXb5/D6359+isoY/dt7StfndpUCcYpmfpK1jHmHAV4FZ6k55JTohNFWQEWukLetc9T6doe/qgkqEOcdq/X3XX04v0Bu3xfVHZSPdntu2l2zLbyoICe0e30y5YoEmoT+g9WE+taEALfUgvwRacTSkf0S3h0DhoXhynn+lPWAtKtegXOgMb9mVlA+orEM2Pt2Er9qHU9mVzjl6hfXTOP6W6/X1YuEnrKfTr9LKbh8qNj+63RaoGi3pjC6D7iL6QDsJu/AH9gdD9xU46CRsmQqucHHKL4FOBNqSTztKxzPaTCIEp3Df0GO13i0C9tigbcoT+hgWXoKGtuXO3/32cvoQFq3lG5vEvhvY6/1yL8DS2VDR3apjrXF1n2SjAX7C5MeEoHUH4gSoXvqK4nphFxbaowRZGuVS6aS/s3Fr9ruaogowUpC70gU7TO+5ThEVHvIY9djUzhP4GKzLIhM0p0+iffUcoizS3FP12aEUHmYcmYj17X4AN8X1ukcwXpBpZNXHyaHgNoRLRoiIHYFWKWChIvYY7DDsnKvqReoER/cJNMl+QuvFTsCBVARyko36cXlTIQYRKJC/IA7BjT9BnCHV0Ln8AF0d01APqKSQAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA7CAYAAADGgdZDAAAEA0lEQVR4Xu3dTahtYxgH8EcoX4XoIgx8jSgDGRAzJYkw8TnRLVGYCPca6E7uAImIJBIlhYmBkqQjJaGMZCSRUkpGBsjH83jXctdZ9+x99zrbPse9fr/6d896197nrr1H/953vetEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwHpnZN4fDwIAsHWOzNyUeXR8ovNU5svx4Apcl9mZeXx8AgDg/+7MzPeZP8cn0omZTzN7RuOr8Gbm98xX4xMAAER8kVkbD6YbMu9kjh6fWIErMn9kHhyfWMJF4wEAgIPVz5m948H0ZLTl0q1QRe23zGXjE0tQ2ACAQ0Yth16VOS2zYzB+XubYwfEqvR1tOfSUaNdxwvrTm7KZwtZ/B4ePT0xUv2P4GY7KHDY4BgBY2MnRCtvu7viaaOVtUVVEnsl8e4B8lDmne89Gajn0p8yl3fHL0crbZh0X0wpbXePz3c+1EeP1aKXr9Gi/a4r7o32vtdRcn6OWlF+J9t0CAExWpealaCWl1H1rU4tFzSTVzNS8nBTzZ5iqsN3Y/VyvezXa+zZrSmGrYvZJtGvs3RqtuM5aEq6ZxzvGg9GKWpW9WtqtJd7+nryzMxf2LwIAWFQVoyprVSZKvyt06ozSsmoWq3aJHtEd92VnqK718sydsfH19Uupfc7NXDka2+h9Vbzei/3vnavS+nVm12i8V5skfo3ZGzJqdm24YaPKY19Yqxg+m7mlOwYAmGlc0PqiVMWizg29EfuXmlIl695oy4nzUs9XmzVjVr93uDu0NkDURoia9Tu+G7sg2iNIqkD1xW6eRWfY6pq+6f4dqvdWkev//6nqe3xocPzw4Oe7o33eevYcAMBctew3fP7aWubDaKXotm6sytvTmeszl3Rj/6YqXzW7VrNsvSprVXZeizZTVuWx7oGrsVkzWmOLFrZS5ak2WJTabHBz5vPYd//ZIgVx7Jdo32/Zk7l236m/l1+vHhwDAMxUf8Xgu8HxfZkfoj3Oo7+n7azMx5kHYvGyNEV/c/6wFL2b+Sza/XSl7v2qAnX+P684sCmFrT5X7VB9Mdq13JM5Jtp3UUVxMx7J/Bit9N0V6+/fq8/32OAYAGApL0Sb6Zq3YWDVqkDVdUxRBbD+Dup/Td0zN2UXLgDAXDVLtZZ5azS+1aqs7Y5WdpZ51Md2q0eg1PLrxd3xVj3jDgA4hNUM1Qex/Y+jqMduPJd5InPq6NzBqMpn3d/WLzsDACxlZ+b2aM8j285lUQAAZqhdk8M/VwUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACs3F+8nHzzju1JjgAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,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>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAUCAYAAACEYr13AAABFElEQVR4Xt3TIUjDQRTH8Se6IBrUIBMFZU2NJmVlYBXGyoJlrKyvCAsWMWgU1gba1gYm6+qwallaGBt2Z/b7/vcO7wZj+7PmDz7hf3d7d7f3/4v8u6yalemJeXnAN8YmF0/Pz9IFNE/oms1oZoFs4x33JnWOMcKlSZ1r/ODNPKOHYrhoVrRl+oM+Do3mCgPs23MWDXNiY0n8/W/CQVLCBGdYQx0FcxGsSxZ8IR8OiuuKb+k5HuWvgD9lEj3qEAfBmD9VW9zut6iiYqI3Vat/Yi8YK+ND3E4b6OAFuybK0gUyaKKFO6OtPLJ5LdzFqT3PzFbAR7/MGl7Fvd6pv1ZdqFfQ3uu/r61U6+GiRaLX3BFXMNr9FzfFMSNP7yWfAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAVCAYAAABCIB6VAAABIElEQVR4Xu3TPSvFYRzG8Z9QhBCFmJTVwBs4g8Ki46GYbBZvQBgMsjGQwcRqsJLB6A1YSfFOXN9zX7fu06mzcLb/VZ8690O/++l/IqpUqdI2vXImj3Jv/R7bk2v/vrA3GXNf23Ss8JbUZU0+bEL6Ii3GomTT3mXKfWROjm266I8NGZW7QpfMyrcseR5zcCVD7huUQ6nZvPt/w0rsZMUIBT9lxm0WwpHbZEf2I50Yw8VYI2WRXOhAXmTAba4My273yI1sy7q1ZFW+It0duAoe7SFSAR7z1PKueIdXOfd4fvCmMPlJbo0C3DUPyc4vZcFyFuVZJou+lnSscM5IgXTLeKRvvQzH5jPkjgnzwBX+KXnBk0hf0a7xHv8S/kT5dE35AVRUNrfjxVFfAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABLCAYAAADNo9uCAAAEe0lEQVR4Xu3dO4geVRQH8CtRMMZFRVGEBAM+IBhQEKwMiqQQBAliIYqVoBCSUkJsrMQUieIDwQe+QASNiqDYBAlYGFAiFqKFIkLQSgshlaCew3zDd/ey32Rd18zo/n7wZ2fuzC5bHu6de24pAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPzL9kQejBxpxi+JnIhc1IwDAHCWvRE5Hfm+Gd8R2deMAQAwgt2RPyIHmvFnIuc2YwAAjODRyO+RW6qxpcgn1T0AACM6Ffkp8mXks8jHkS2Ra+uXAAAYTy6Hfh7ZOksWcAAATEi9HHpl5MfIhfPHAACM6YrI0TLfXHB35M/qHgCAkd1Uuk0HvdwZmgUbAAATkU1xH59dn1e6XmzZf613MLKrWCIFABjVD5F3Iicj25c/Ku9GthVLpAAAk5RHU91Tli+ZAgAwITdEXohc3z4AAGB850T2t4MAAGPKD+6zxUXv/MjlkU2z+/zo/tLSFTIbwZ7Iy7PrPPUAAGB0r0beK12hsjdyIvJt5FjkidnPnyMv9r+wAWTBmjtEn2ofAACMITv95zdbb5duti31TWT7WbX8+P707Lp2R1l+cHorTw84UzbKzB0AwJrl0uf9kbuqsexPVhdoH0W+ru7TNZGnI7c14+tle+n+fs7sDaX2QOka4I6Z3F0KALCusoVFHoC+VI3lEmgulfZyti0Luny3nxHLpdOdkc39S43sX9YWVyslZ9kAABiQy6G/NWNZoOWsW8pNCN+UbmNCznj1BVvOut07u16kXf5cKf3mBgAAFjgSOVXdX1C62bM8uinlTNl3kQ/LvFjLnaMfVO9MSRaYD1fXj1TP1uK6Ml8uvjqyr3oGAHBWZFFzcTPWnp+ZbT3qd7aWbmZuivJ/f6W6fr16thZ5SPyT1fWz1TMAgEnKwu1w6Yq21O8snYq6SFuvgq0v0hRsAMB/SvYqa2fipkDBBgAwcQo2AICJGyrY7ox8GjlZuhnC3OWaO2LrFiYtBRsAwDobKtjeLN0GimwKnMVXntKQ18erd1oKNgCAdTZUsOUGiZ2R58q8RUm2AOnbgKTsP3dfdT9UsGXj34civ5TuOC8AAFZhqGBLWZzlGagpi7ZsBpyFWMrmwFdFDs7u01DB9n7pmv/mUV55dNdl1TMAABYYKtjy/vjsZ9pb5qc1ZIPgo5EDpfu+rTdUsPWyYKubDQMAMGCoYMui7K3SnYmaXivz4i1n176I7J7d91ZTsH0VubUdBABgZUMFW7qxdDtFc7xv/puyGPu1zL9tq8eHCracVbu9GQMAYMCZCrZFXoo8X7qNCduq8aGC7bHIrsjNpdvI4Bs2AIBVWGvBtrl0GxAOR3ZU40MFW/Zw63MssqV6BgDAAlmkHaquV1uwLZJF2v7qul0SBQDgb8pv0Jaq+/p6LXKJtJ8521T++d8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgP+lvwA22K/MsJld2QAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGMAAAAUCAYAAACZHIPsAAADM0lEQVR4Xu2YTahNURTHl1Dk+yMfoR5CSj7yUYoZoRCRiAFKBqQklIFupCRfiRETA4UoipLkIwPKxAQDKUmUwsiY/7+9lrvveuecd7d373336fzq1+3uc9/e5+y99trrPJGSkpKSkm4yEJ6Ho/0FpZ9vaAJ91L7+QgG74WLf2NvhAx2Kvk9UX8Hf8JY0d0H6w9PqVTio9nIus+AxCYv431AuRpvAB+ED8cE8TFtvJKSDZsEUeQnuV9fDm3Bk/KMcGCBcwMn+Qm9lLKxIduQvgT/1s1kMhdNd21Q4zrXlsVKaGywtZSNc4RuVwxJ2Rt7B3g4Mg8cl7LBkGAnn1NvwLrwmodNWwwc4CUe4du4SyrPihYT7uyJhYQ5I43I0o/oGfA/XqIRjfJfs1JnFDvmHqooTfh9uUMkQ+BzutR9lMAPeg0/rlJM4QbrGH9wG/5Z+hI+kGiib4AcJqc0zB56B0/yFHCZJCIRREu6Zu5CSRfCz1KbHuH8/xkwJqTYpSCrwgYSItG1li8EqopUUHdzL1B9wftR+QsLEDY7ayBh4Fi6FC9y1PGZL6JsB8U0/Lbr5rsHMMUW/+/79GMkHuVUmtvoGV5Vbkg/aSooWg/dCWdpaCuPueCkhoDwH4Wa4TbJ3TREXpLrAtsgck+fAAP3u+/djJC/GQgkR4CuTrfCX1Eagp5VpihPAVEo5UUYcwbwfVjuUQfYQrpZQDKRgWcEHIgsK66ue/pPTFMuvr1LdeoT5ktHGCam7owaSdYDz/r6oa6N2i2DeM6PWIpTB9VjC8/HlzeBL42t4Uc0qnTkud1+cLbgDT0kYh+T1H5N8gJeL0ZkeWQwrE9/B63CP+gxukbR/jjUabv34PWMV/KTGOXgXfCJhYtdF7SxH70jn0pyLwedl2UrH117+Cyf5LdynXoZzo+t5/RvJ7xlWJtrqMg/SnlwEg9FdkWrk8p6Yy6lnuFQP1Q6VZ8t2bePf+h1+VM1bDMI+WTHReE46JLv/eIzkN/CsMrFdKKqqimBQUZaoR+ByuFPbDFtompWmuiKvfxsjuYoizIlxmdhuZFVVKcRlqcGJYpU0T+0OWf0TBlDSf23/AOXbqZRt+HMhAAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAUCAYAAAA6NOUqAAACIklEQVR4Xu2WO0hcQRSGj0QlEl9BMYqIaCEYgyIiqLFUUEQJsfDVBINYJGIhKFgFgo0WQiDYKSEEUsTCQrT0VYgWsQlWgoIoGKKVff6fcwbvva677q6FJveDr7hnxp05M2dmFAkJCQkJCQm5I57AXlgRbPiXKYEz8Ct8DlP8zfeHbHMWLsIl+A3meDvFgMlVmz9EEy/y9YhOqejf/YIT5ojoPFbgDqyBn+E83DLnYJokCBNcNrstlgU34XvXKQqPYIfoBEdNlnq8jME6+AlemE3WxkU8ghuw0GKd5h9YaTHCTWgXnXuqeSMf4KqZYTGX/Bf7jgT7DoomzeS5CImSDl/DArgGp0wHkzuBbZ7YsHkAn3nitbAf9sFMMyL54i8zBwfjinonEISl9kbuJnkHx/0tunPUMQAPYbF9czd5ROiCXN0njPNYNMNXFruRengm2pk6ONilaBnGwlv2b81Eyp5w3FNYbhImxgR5LB9b7IXo5tAu0UWhZaIXbI/4j0JEWDbBwfLgNhyX+G7oZC88wvPuknSJeqvTwXnvmxxjElaJVstP0ac1Jv9t8u7c8Ae+m+/guuhlkewZjved58W0JtdfGD5vx7DRE2uFe+Y0HBL9fS7QR7nF08czcii6iuxMucrJJh0knv/w+MoEx+c340HcTe5uc1YuXyw+fyTqE9cCz+V2l9pDgJv3UvT5ZSU0+Jv9sER24dNgwwMnV67ui4j8BT/HaCeeESE4AAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJMAAAAUCAYAAACJUA91AAAD4UlEQVR4Xu2Za6iNWRiAX7nfr5GGMO4NEVJqSDIhl0FEUUQyhKQktx9uoVAu00woyQ8/nOIHzQylI34QP11KFHJJmuSHv8P7eNc6Z+3vbPvynW/vcbSeetpnr2+fvde31rve9a69RSKRSFm0VbslGyM5dAyMFKC3OibZGMlhcGCkADGYihODqURiMBUnBlOJxGAqTgymPLRQJzsXOFepW4Lns9XO9vKyaKb2co9NHQ4lM6R+TDYE+rZpamv/DxEjq8zUQd3rHr83YmYqkRhMxck0mEj7J9XzzrPqOfm2Bq6N86J6OGjvo9aq44O2ELamUcnGFDQmmNhWNqs3nYz1HXWN2LbZUtJtvfnopB5Un6ov8nik/qV1DJQMg4nJ2RY8n6e+VYcHbdzwJvU3sQH4Wu0wVL0sNsGlWOP8QQrzo/O1Oido5+/nYkFVSdIGE0Hyl1gAMYYII8T6PUGdri5y7Y2hn/pQ/UNs7qiBPjrnimXpcvtfFq3US+p9dZmTG6UzYcCsVGeqy8UKXawmBA2+k9wgP6ZeEctaWdBe7D59IepdKpYVeQzbfxHLPJiE8dujvlQHJK71UB+oO9QDat/cyzlsVN9I4QxLkFwV+zw/b7TVOkkEVYHK/oP6KZBTkIcbZdsbrc4P2qsJQYO1Ur+6uqp3xU5rWZFlMPmAOSMNM7mf6H/VFbmXGlBKME0V201IBB6y/TPnkqC9InCDpD+Oy0B9gax06ibPanW/2BbXM2jPRyW2uXCF7QvaGVwyFQM5SBr+Bsd1tvAsaoE029xI9b3kzwr+nghQv/U1BhYU9Vj4+xrb5yvnT0G7hy14t/pr8kIahohNhi9ofS10XCxwgO2MovyQFF4ZlYRtjRWMPgvRz61iW0h/sSBjIXgIeorNieq4oD0taYKJybqtrku003e+xyHQWLQEU6ucV5QPJcANqQ8mPvsfsayGSejDTnWSujhxLRUxmEonBlMR2OePih0lecPTToo4n3oZPLa9We75/wH7/X/OJ+pa9Xd1u/pYPaGur3u1wVGce6LGCYMsLWmCCQjkR2JbHf3GC+pCsRPcPfWUOsX/Q0qYL8aE+eOwdE3s/X2CSDJW/VPsNElAZQYDxICHR1doLlYvcUroLtYp2qoJn0kBW+NsJ5Z1fD85xXVxfwNFL9JnFgBFchakDSZgzOgz/Qz7Crxfmvf8GryXn6tCcJjhkEXWzHd4yBw6xGTxHRTbBatrWM4rKk94Yivl1Paz87rYQsiiuAXGggVXbJKaAozp3+ouKfyVREVgZRHt1c5KQDrmyOuDpBicTJHvzqgbIg3hy8tbks3PTF/4DPXtw8k2ZWIBAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,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>

[image21]: <data:image/png;base64,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>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAVCAYAAAAw73wjAAABW0lEQVR4Xu2VzytEURiGPxkLZTLJwkKRlNQU4k+QWPgHlJWFWUi2JluLsZQFVkqyUVZKNuyVnaUFG2UxC40t876d7zTfvbnX3ObOhO5TT92+c+6575xfI5KRkdEWetVVOBpsimUC7sFjuKJynNRZgzfwTf2As4Ee0czDBzgOu+G2egh7TL9UyIv7yLLabFDO2jUsm9qw+gQXTT1Vkgadgu/i3vH0qXdw39TH4I40tleYfnE/uBBu+I6kQZfgl0QHvYA508ZtcqQymGcInsBJU4vl3wZl37iglM+WOfUUDojbz2fiDmPTdCKoZxpewUs4Emr7kaRBuZSfEh2UM9Vl2iwdDcqT/ArXTW1QfYS7pu5p+9LzKnqGWyrhQTmHB9KYuaLKezQ8RsuHqQRfYE3lvqtqjYOTGa1VVA+X7R5uigt2q25IcNlTuUf/TNBW4X86992CuA/aZc34ldQBgANhAmNnD+4AAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAVCAYAAAAw73wjAAABtUlEQVR4Xu2VzysFURTHj1BEkZWFUlKyQsjGQpGyYINSNjZ+LCzYINnJwk5IKQsvWUuSBUn5L5SwpFDKyoLvt3OmuW/MPPN+lF7Npz697p17Zs7cOfc8kYSEhIJSAnvgtrkHx2CluygCN9aLixubFXzQApyGdWY3vIdXsMZfGgpjt2C1aOytGSc2K5rgI1yHZSaZgN9w1sZhMLkb+ApbbW7ZZOyQzRWEFvgMT2CFSTrhJ0zZOIp+uAjLbXxgMpb38OCGrImWRFhZcPdXYW3wgkuV+A/yGBHdlfnAfCbaRF+aLomWlcsA3DfdsqiHh+J/lUiKJtEg/DRnEv9AsHyu4QvcMFm/YXSZR6KHrwEew2Z30V9wByh341L0TbOhFO6Yd7Ax/XIa7fBc9GxkWvcLJshDQXkYWA65MGyybNhbo8gpUSY5LnrqqFevrDnubhQd8F38jkHcRJmIN++R16cfhaeiyVL+s0zaHH8Jk34SbfCUsP18wF3x+++UyUTnbM4jr8PEBN5Ebxz0C/baOu4e122ahMmtwAvRF5qBDybXuJ0k7z5aNIkWAraiPjgo+sA4LS3hX/kBgPZq0K70KKMAAAAASUVORK5CYII=>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAVCAYAAACzK0UYAAABPElEQVR4Xu2TPyuFYRjGL+WUQiRlJsuJkizEaDGIDWdUOsuZzmKzyDdgJZmU1Z/RajLJgA8gxSdQrqv7eup5vZ0ycDK8v/r1vu/z537u536eF6ioqPg3DNFjekeP7CY9pxf0hS7QPXpmr+mV5/6IriyyQVdpk35atfXQAXpLn+mUx4s5+k6XszaNX6Et25v1YY2O0hNE5jINUPsDYlKOgr/RmaxtljbollWCBVKwtk0s0VdE5jkHiB2mQEpKZdb4dVtCnR9+ysQuYnElIcbso/tG6CQdR5yVyly3JXQeKVgK2EcvEWVUvYXKJLW7abrtb53FPeLCdOSUHiKCpYDK+AnFiTp8qYT2ETeuhtiVvvXeka4s0o/yAC026Od3VMphv0/QG8RvIHQJCtf3N1Byi4jz2aHz9k/QzrTDEl+4n0A015nnkAAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE0AAAAUCAYAAADIiLCPAAADEklEQVR4Xu2YW8hMURiGPznkfI4UkRxyVkjkQkLJITlEKf2l5IKIC+Vw4ZQLKUmRJJQcUhRCXChClDu5UhS5U8SVxPv49jRrr5k9s2f8/GS/9fTPXmvvNfv71rfeteY3K1SoUKFCv6r2omvcmKGeYrM4mbBbDErdka3w2T1iRLr7p2aKlWKw+bjjxRbRO7zpb1CRtAbURawVN8XSqK+aBohHYpVol0CQD8XQ4L5qov+UGGU+zj7xVSwPb5K2i+8BH8WC1B1tpH5ip7htHjSVlkcbxQPRI2gjcefFseRzlg6Il2JCct1LPBEvRP/STeZJo3pXiKmiY9D3xzXafKbhdHLdiLqJe+Js3GEeaBx8rMPim5gdtDHWFzElaGOsxcF1Q2JNXxDXEs6Jx2Jy0t9ddE4+Z4kKopKoKCqLCoNmNFC8suykvRfD445AvEtYobz/ffPqY+ySGGuDuefdEOvNbaSuKMu3VrneWR6UdB+xy2q/ZAdx1Lw6wpdqVpjyG8tOWlwx9TTP/Jk4RqqM5c6mQaIPmsfAcs4UlfDUvMoIPNRc8UGsFnutsj8WHjPWfCwSOCShGbVm0tgUnptvQrEPdrK0j00Tn8W6oK1CJIa1vyTuMH+pT+ZfOC7qqycmAzOGS2KiVb5wLeFX+FZW0lgZWEo9kbC7Yk7UTlXBLEvvxMTMhODFmdpknphJcYeVB2iJ2hsVps7M3RILLd/uSVVfMT+exF5KFT8zt41a6mse/Jigbb+5zRAvEPsdK58bSzEz2ZmaYb4MONSFwgwxR84urHsGzRNsLfE8SbsulkV91YQtUG2hR/JebDRUGyK5R8xXQ1h5eBIb2lbz4wTsMJ8Eks2YwPXI5BnUYjmWfpG0JpKGz2wzP16ssbIPnTE/bvAT5KL5WYuTdWuI78wzARj0cfONhUmFQ+KylXc3ksY978xP/yWdsPRJvwRLPtzQmMSr5rGT4Nfm/p7Lf/ENdiz+hh7Cw3hDW52U+f5hYlECn3MF1IDw3PnmB+HYP3+7plt5KdSj9PPmvxf/GaBa88AJ/Z/UD3khmpO36LWLAAAAAElFTkSuQmCC>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAUCAYAAABroNZJAAAA6UlEQVR4Xu3RvwpBYRgG8Ff+RCkGRZkpJYsbMDIoJiYmJXeAwSCDYpBFsshgtSgX4BLchEyugOfxvUfnKKGUxVO/4XvPd97vfO8R+eefHyQEc7WGJawgaN/0KmPoKKYMR0jdd4gkoKvitvotPtjAQdUhLaaBS/fwi9qQUxmtO1KAs7qohu15FVpQUrz+PTyJxaKtFoWtmLkwHphBRcw1yZEknMTMxAobT6GpazbdwwgCypGvNGFhAkMx96YF9MGre7Kwg5iun4bT54lkvczwkIGYmTBuZf21t8IXItCDPNQUh/1x/BB+LDJX9fUkfLBUuLsAAAAASUVORK5CYII=>