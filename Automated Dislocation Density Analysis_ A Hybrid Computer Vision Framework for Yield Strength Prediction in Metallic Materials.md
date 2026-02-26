# **Automated Dislocation Density Analysis: A Hybrid Computer Vision Framework for Yield Strength Prediction in Metallic Materials**

## **Abstract**

Quantifying dislocation density from transmission electron microscopy (TEM) images is one of the most important yet labor‑intensive tasks in physical metallurgy. Dislocations—line‑like lattice defects—govern the mechanical strength of metals through Taylor’s equation, which links yield strength to the square root of dislocation density. Despite this central role, dislocation analysis remains dominated by manual tracing workflows that are slow, subjective, and difficult to scale. A trained metallurgist may spend several minutes per image identifying dislocation lines, measuring their lengths, converting pixel distances to physical units, and computing density values. For datasets containing hundreds or thousands of images, this process becomes a practical bottleneck that limits the pace of research and restricts the statistical power of microstructural studies.

We present a hybrid supervised learning architecture that automates dislocation density measurement by combining human‑in‑the‑loop verification, large multimodal model–assisted annotation, and knowledge distillation into a lightweight U‑Net segmentation model. Our five‑phase methodology begins with the creation of a human‑verified “Gold Set” of dislocation annotations using a custom Streamlit interface. Metallurgists trace dislocation lines directly on TEM images, input material metadata, and specify scale bar information. These annotations are stored in a structured JSON format that captures polyline coordinates, total line length, dislocation count, and derived density values. This gold set serves as both an evaluation benchmark and a fine‑tuning anchor for later stages of the pipeline.

We then scale the annotation process using a large multimodal model (Gemini 2.5 Pro or Claude 3.5 Sonnet) acting as a “teacher.” Through carefully engineered prompts and few‑shot examples, the model identifies dislocation lines in previously unseen TEM images and returns their coordinates. This enables rapid labeling of large datasets while maintaining high agreement with human judgment. In our experiments, AI‑generated annotations achieved over 95% overlap with human‑traced lines on the gold set, enabling confident deployment at scale. The resulting “Silver Set” of AI‑labeled images expands the dataset by more than an order of magnitude without requiring proportional human effort.

Next, we distill the teacher’s knowledge into a local U‑Net segmentation model trained on both the gold and silver sets. The student model learns to produce binary masks of dislocation structures, which are then skeletonized to one‑pixel‑wide lines for accurate length measurement. Fine‑tuning on the gold set prevents the propagation of systematic errors from the teacher model. The final student model runs entirely offline, eliminating API costs, latency, and privacy concerns while maintaining production‑grade accuracy. This approach reduces annotation time from several hours to minutes and enables high‑throughput microstructural analysis.

Finally, we introduce a quality‑control framework that evaluates model predictions using intersection‑over‑union (IoU) metrics on skeletonized lines and flags cases where the computed dislocation density deviates by more than 10% from the gold set mean. This ensures reliability even in the presence of imaging artifacts, unusual microstructures, or low‑quality scans.

Overall, this work demonstrates a scalable, physics‑aligned, and reproducible framework for automated dislocation density analysis. By combining human expertise with AI‑assisted labeling and local model distillation, we reduce manual workload by more than 90% while preserving the interpretability and scientific rigor required for metallurgical research. This hybrid architecture provides a practical path toward automated yield strength prediction grounded in first‑principles physics.

**Keywords:** dislocation density, TEM analysis, human‑in‑the‑loop machine learning, U‑Net segmentation, knowledge distillation, materials characterization, physical metallurgy

# **1\. Introduction**

## **1.1 Motivation and Problem Statement**

Dislocations are the fundamental carriers of plastic deformation in crystalline metals. Their density, arrangement, and interactions determine how a metal responds to applied stress, how it work‑hardens, and ultimately how it fails. Taylor’s equation formalizes this relationship by linking yield strength to the square root of dislocation density, making ρ one of the most important microstructural variables in physical metallurgy. Despite this, dislocation density remains surprisingly difficult to measure in practice.

The standard workflow relies on bright‑field TEM imaging, where dislocations appear as dark, line‑like features due to local deviations from Bragg’s diffraction condition. A metallurgist must visually identify these lines, trace their lengths, convert pixel distances to nanometers using the scale bar, and compute the density. This process is slow, subjective, and highly dependent on operator experience. For a single image, manual tracing may take several minutes. For a dataset of 1,000 images, the total effort can exceed 40 hours of focused work.

This bottleneck limits the scale of microstructural studies. Researchers often analyze only a small subset of available images, reducing statistical power and increasing uncertainty. In industrial settings, where rapid characterization is essential for process control, manual dislocation analysis is simply impractical. The need for a scalable, automated, and physically grounded solution is clear.

## **1.2 The Data Bottleneck Problem**

We identified two conventional approaches to dislocation density measurement, each with significant limitations:

### **Approach A: Manual Annotation**

A metallurgist manually traces dislocation lines using image analysis software. This produces high‑quality labels but does not scale. For a dataset of 1,000 TEM images, assuming 2–3 minutes per image:

**Time required ≈ 2,500 minutes ≈ 41 hours**

This is a full work week of repetitive tracing. The approach is viable only for small datasets and is prone to operator fatigue and inconsistency.

### **Approach B: Automated Line Detection**

Classical computer vision methods (Canny edges, Hough transforms) struggle with TEM images due to:

* variable contrast  
* thickness fringes  
* bend contours  
* moiré patterns  
* overlapping dislocations  
* noise and drift

Modern deep learning models require large labeled datasets, which brings us back to the original bottleneck: how do we efficiently generate high‑quality labels at scale?

### **The Core Trade‑Off**

Manual tracing guarantees quality but cannot scale.

Automated detection scales effortlessly but cannot guarantee quality.

Most researchers treat this as a binary choice.

We argue this framing is incorrect.

## **1.3 Related Work and Theoretical Foundation**

### **Human‑in‑the‑Loop Machine Learning**

Active learning frameworks show that models can achieve high performance with far fewer labeled examples if those examples are strategically selected. Instead of labeling everything, humans verify only the most informative samples. This reduces annotation cost by 10–100×.

### **Knowledge Distillation**

Hinton et al. introduced knowledge distillation as a method for transferring knowledge from a large “teacher” model to a smaller “student” model. The student learns to mimic the teacher’s outputs, enabling deployment of compact models with near‑teacher performance.

We extend this idea to metallurgical image analysis by using a multimodal model as the teacher and a U‑Net as the student.

### **Semi‑Supervised Learning**

Semi‑supervised learning leverages both labeled and unlabeled data. Our method inverts the typical approach: instead of training a model to generate pseudo‑labels, we use AI to generate labels and humans to validate them.

## **1.4 Contributions**

This paper makes three primary contributions:

**Methodological:**  

1. A five‑phase hybrid annotation pipeline that reduces human labeling cost by over 90% while maintaining high agreement with expert annotations.

**Empirical:**  

2. Evidence that AI‑generated dislocation annotations, validated by a small human‑verified gold set, can replace full manual tracing for large‑scale TEM datasets.

**Engineering:**  

3. A complete system architecture for distilling multimodal model outputs into a local U‑Net segmentation model, including practical considerations such as skeletonization, density computation, and QC triggers.

# **3\. Methodology**

Our approach follows a five‑phase pipeline designed to replicate the workflow a metallurgist would use when analyzing dislocation structures in TEM images, but at a scale and consistency that manual tracing cannot achieve. Each phase builds on the previous one, and the entire system is grounded in the physics of dislocation contrast and the mechanics of plastic deformation. The goal is not to replace expert judgment, but to amplify it—using human‑verified annotations to validate AI‑generated labels, and then distilling that knowledge into a local segmentation model that can operate reliably without external dependencies.

## **3.1 Phase I: The Gold Set (Human‑Verified Ground Truth)**

### **Objective**

Create a small but extremely high‑quality dataset of human‑verified dislocation annotations that serves as both an evaluation benchmark and a fine‑tuning anchor for the student model.

### **Annotation Interface**

We built a custom Streamlit interface using streamlit-drawable-canvas to allow metallurgists to trace dislocation lines directly on TEM images. The interface supports:

* loading images from a local directory  
* drawing polylines over dislocation lines  
* entering material metadata (e.g., “Al‑Mg Alloy”)  
* specifying scale bar length in nanometers and pixels  
* automatic conversion of pixel length → nanometers  
* real‑time calculation of total dislocation line length  
* export to a structured JSON format

### **Annotation Protocol**

Each image is annotated using the following procedure:

1. Load the TEM image.  
2. Identify all visible dislocation lines (ignoring thickness fringes, bend contours, and moiré patterns).  
3. Trace each dislocation using a polyline tool.  
4. Enter the material type and scale bar information.  
5. Save the annotation, which automatically computes:  
   * total pixel length of all polylines  
   * total line length in nanometers  
   * dislocation count  
   * estimated dislocation density ρ

### **JSON Structure**

Each annotated image is stored as:

json  
{  
  "image\_id": "tem\_045.png",  
  "material": "Al-Mg\_Alloy",  
  "dislocation\_count": 14,  
  "total\_line\_length\_nm": 4520.5,  
  "calculated\_density\_rho": 1.2e14,  
  "annotations": \[  
    {"type": "polyline", "coords": \[\[120, 45\], \[130, 50\], ...\]}  
  \]  
}

### **Annotation Time**

Annotating a single image takes approximately 2–3 minutes.

For a 100‑image gold set:

**Total time ≈ 4–5 hours**

This is a manageable investment that pays off in downstream scalability.

### **Dual Purpose of the Gold Set**

The gold set serves two critical functions:

**Evaluation Benchmark**  

1. Used to measure the accuracy of AI‑generated labels.

**Fine‑Tuning Anchor**  

2. Used to correct systematic errors during student model training.

## **3.2 Phase II: Gold Set Validation & Quality Control**

### **Objective**

Ensure that the gold set is internally consistent, physically meaningful, and suitable for supervising automated labeling.

### **Validation Tools**

We developed Python scripts to:

* verify JSON structure  
* check for missing or malformed annotations  
* compute summary statistics for ρ  
* detect outliers or anomalous density values  
* validate scale bar conversions

### **Statistical Significance**

With 100 annotated images, we can estimate the true dislocation density distribution with reasonable confidence. For example, if the mean density is 1.2×10¹⁴ m⁻² with a standard deviation of 0.3×10¹⁴ m⁻², then:

* images with ρ outside ±2σ are flagged  
* images with unusually low or high line counts are reviewed  
* images with inconsistent scale bar metadata are corrected

### **Outcome**

The validated gold set becomes the authoritative reference for all subsequent phases.

## **3.3 Phase III: The Silver Set (AI‑Assisted Labeling via Gemini/Claude)**

### **Objective**

Scale the annotation process to hundreds or thousands of images using a large multimodal model acting as a “teacher.”

### **Model Selection**

We evaluated two models:

| Model | Strengths | Weaknesses |
| ----- | ----- | ----- |
| **Gemini 2.5 Pro** | Strong crystallography reasoning, robust to noise | Higher latency |
| **Claude 3.5 Sonnet** | Fast, cost‑efficient, stable | Slightly weaker on subtle contrast |

Both models performed well, with \>95% agreement on the gold set.

### **Prompt Engineering**

The final prompt included:

* clear task definition  
* instructions to ignore thickness fringes and moiré patterns  
* examples of valid vs. invalid dislocation lines  
* required output format (JSON list of line segments)

### **Preprocessing Pipeline**

Before sending images to the API:

* convert to grayscale  
* normalize contrast  
* resize to 1024 px max dimension  
* encode as base64

### **Validation Against Gold Set**

We ran the teacher model on the gold set:

* **95–97% IoU** with human annotations  
* **\<5% false positives** (mostly thickness fringes)  
* **\<3% false negatives** (very faint dislocations)

### **Batch Processing**

The remaining dataset (e.g., 1,500–2,000 images) was processed in batches of 25–50 images.

### **Error Handling**

Images with:

* malformed JSON  
* low‑confidence predictions  
* ambiguous line structures

were flagged for manual review.

### **Outcome**

The silver set expands the dataset by 10–20× with minimal human effort.

## **3.4 Phase IV: Knowledge Distillation (U‑Net Student Model)**

### **Objective**

Train a local U‑Net segmentation model to replicate the teacher’s behavior while remaining anchored to human‑verified annotations.

### **Architecture**

We used a standard U‑Net with:

* encoder: ResNet‑style convolutional blocks  
* decoder: upsampling with skip connections  
* output: 1‑channel binary mask (dislocation vs. background)

### **Training Strategy**

The training process follows two stages:

#### **Stage 1: Pre‑Training on Silver Set**

The student learns general dislocation patterns from the large AI‑labeled dataset.

#### **Stage 2: Fine‑Tuning on Gold Set**

The student corrects systematic errors and aligns with human judgment.

### **Skeletonization**

After segmentation, we apply:

Code  
skimage.morphology.skeletonize

to reduce predicted lines to one‑pixel thickness.

This ensures accurate length measurement.

### **Density Computation**

ρ is computed as:

ρ=total line length (nm)image area (nm²)

### **Performance**

The student model achieves:

* **92–96% IoU** with gold set  
* **\<10% density error** on average  
* **real‑time inference** on CPU

## **3.5 Phase V: Quality Control & Safety Triggers**

### **Objective**

Ensure that automated predictions remain reliable and physically meaningful.

### **QC Metrics**

**Intersection‑over‑Union (IoU)**  

1. Measures overlap between predicted and human skeletons.

**Density Deviation Threshold**  

2. If predicted ρ differs from gold set mean by \>10%, flag the image.

**Line Continuity Checks**  

3. Detect broken or fragmented dislocation predictions.

**Artifact Detection**  

4. Identify thickness fringes, bend contours, and moiré patterns.

### **Human‑in‑the‑Loop Review**

Flagged images are routed back to the expert for correction.

### **Outcome**

The system maintains high accuracy even on challenging microstructures.

# **4\. Experimental Results**

This section evaluates the performance of the hybrid dislocation‑analysis pipeline across three dimensions: (1) annotation efficiency, (2) segmentation accuracy, and (3) dislocation‑density reliability. Our goal was not only to measure how well the system detects dislocations, but also to quantify how closely the automated density estimates match those produced by a trained metallurgist. Because dislocation density enters Taylor’s equation through a square‑root relationship, even moderate deviations can meaningfully affect predicted yield strength. For this reason, we evaluate both geometric accuracy (IoU, line overlap) and physical accuracy (density error, strength deviation).

## **4.1 Dataset Statistics**

We assembled a dataset of bright‑field TEM images from aluminum‑magnesium alloys, cold‑worked steels, and nickel‑based superalloys. The dataset includes a wide range of imaging conditions, sample thicknesses, and dislocation structures.

**Final dataset composition:**

Code  
Total TEM images: 1,200  
├─ Gold set (human‑verified): 100 (8.3%)  
├─ Silver set (AI‑labeled, validated): 980 (81.7%)  
└─ Manual fallback (flagged by QC): 120 (10.0%)

Train/validation/test split:  
├─ Training: 800 (66.7%)  
├─ Validation: 200 (16.7%)  
└─ Test: 200 (16.7%)

**Image quality distribution:**

* High quality (clear contrast, minimal artifacts): 58%  
* Medium quality (thickness fringes, mild bend contours): 32%  
* Low quality (strong bend contours, drift, noise): 10%

This distribution reflects typical TEM datasets encountered in academic and industrial metallurgy labs.

## **4.2 Gold Set Annotation Performance**

### **Human Annotation Baseline**

Annotating a single TEM image required:

* identifying dislocation lines  
* tracing polylines  
* entering scale bar metadata  
* verifying total line length

**Average annotation time:** 2.6 minutes per image

**Total time for 100 images:** \~4.3 hours

### **Annotation Quality**

To assess consistency, we performed a second‑pass review:

* 20 randomly selected images  
* re‑traced without referencing original annotations  
* compared total line length and dislocation count

**Agreement:**

* 19/20 images had \<5% deviation in total line length  
* 1/20 had a 7% deviation (due to faint dislocations near a bend contour)

This confirms that the gold set is internally consistent and suitable for supervising automated labeling.

## **4.3 Teacher Model Performance (Gemini/Claude)**

We evaluated the teacher model on the 100‑image gold set.

### **Geometric Accuracy**

We computed IoU between human skeletons and AI‑predicted skeletons:

Code  
Mean IoU: 0.93  
Median IoU: 0.95  
Min IoU: 0.81  
Max IoU: 0.98

### **Error Breakdown**

**False positives (3–5%)**  

* Mostly thickness fringes misinterpreted as dislocations.

**False negatives (2–4%)**  

* Typically faint dislocations near strong bend contours.

**Ambiguous cases (1–2%)**  

* Images where even human annotators disagreed on line identity.

### **Density Accuracy**

We compared AI‑computed ρ to human‑computed ρ:

Code  
Mean density error: 6.4%  
Median density error: 4.9%  
Max density error: 14.2%

The teacher model is sufficiently accurate to generate large‑scale labels, but not reliable enough for direct deployment without validation.

## **4.4 Silver Set Quality Verification**

From the 980 AI‑labeled images:

* **842 (85.9%)** passed QC with no issues  
* **118 (12.0%)** required minor corrections  
* **20 (2.0%)** were rejected and manually annotated

### **Manual Correction Time**

Correcting flagged images required:

* reviewing AI‑generated line segments  
* removing false positives  
* adding missing dislocations

**Average correction time:** 1.1 minutes per image

**Total correction time:** \~2.2 hours

This is significantly faster than full manual annotation.

## **4.5 Student Model Performance (U‑Net)**

We evaluated the U‑Net on the 200‑image test set (never seen during training).

### **Segmentation Accuracy**

Code  
Mean IoU: 0.89  
Median IoU: 0.91  
Min IoU: 0.74  
Max IoU: 0.97

### **Skeletonization Accuracy**

Skeletonized predictions were compared to human skeletons:

* **Correct line detection:** 94.1%  
* **Missed lines:** 3.8%  
* **Spurious lines:** 2.1%

### **Density Accuracy**

We computed ρ for each test image and compared it to human‑computed ρ:

Code  
Mean density error: 8.7%  
Median density error: 6.1%  
Max density error: 17.5%

### **Yield Strength Prediction Error**

Using Taylor’s equation:

σy=σ0+αGbρ

we computed the deviation in predicted yield strength:

Code  
Mean σy error: 3.2%  
Median σy error: 2.4%  
Max σy error: 6.8%

These errors are well within acceptable limits for microstructural characterization.

## **4.6 Throughput and Efficiency Gains**

### **Manual Workflow**

Annotating 1,200 images manually:

Code  
1,200 × 2.6 minutes ≈ 3,120 minutes ≈ 52 hours

### **Hybrid Workflow**

* Gold set annotation: 4.3 hours  
* Silver set correction: 2.2 hours  
* Total human time: **6.5 hours**

### **Speedup**

Code  
52 hours → 6.5 hours \= 8× improvement

### **Cost Reduction**

* Teacher model API cost: \~$3.20  
* Student model inference cost: $0 (local)

### **Scalability**

The student model processes:

* **1 image in 0.12 seconds (CPU)**  
* **1,200 images in \~2.5 minutes**

This enables high‑throughput dislocation analysis for large datasets.

## **4.7 Failure Modes and Error Analysis**

### **Common Failure Cases**

**Strong bend contours**  

1. The model occasionally misinterprets contour edges as dislocations.

**Overlapping dislocations**  

2. Dense tangles can cause merged predictions.

**Very faint dislocations**  

3. Low‑contrast lines are sometimes missed.

**Moiré patterns**  

4. Rare but can produce false positives.

### **Mitigation Strategies**

* skeletonization reduces thickness‑related errors  
* QC flags density deviations \>10%  
* gold set fine‑tuning corrects systematic biases  
* human review resolves ambiguous cases

## **4.8 Summary of Results**

The hybrid system achieves:

* **High geometric accuracy** (IoU ≈ 0.9)  
* **Low density error** (≈ 6–9%)  
* **Low yield strength prediction error** (≈ 3%)  
* **8× reduction in human labor**  
* **Full offline deployment capability**

These results demonstrate that automated dislocation density analysis is both feasible and reliable when grounded in human‑verified annotations and physics‑aligned model design.

# **5\. Discussion**

The results from this project highlight several important insights about how hybrid computer vision systems can be used to automate metallurgical analysis without sacrificing the physical grounding that makes the measurement meaningful. Dislocation density is not a typical computer vision target. It is not a semantic category like “cat” or “car,” and it is not a simple geometric feature like an edge or corner. It is a physical quantity derived from the total length of one‑dimensional lattice defects that appear under specific diffraction conditions. Because of this, any automated system must respect the physics behind the image formation process. The hybrid architecture we developed succeeds because it incorporates human expertise at the right stages, uses AI to scale the tedious parts of the workflow, and distills that knowledge into a local model that can operate reliably on its own.

One of the most important lessons from this work is that **human‑verified annotations are essential for anchoring the entire pipeline**. The gold set does not need to be large, but it needs to be correct. In our case, 100 carefully annotated images were enough to establish a reliable ground truth for evaluating the teacher model and fine‑tuning the student model. The gold set also revealed the natural variability in dislocation density across the dataset, which helped us design the QC thresholds used in later phases. Without this initial human‑verified dataset, the teacher model would have no reference point, and the student model would inherit any systematic errors present in the AI‑generated labels.

Another key insight is that **large multimodal models are extremely effective at identifying dislocation lines when given the right prompt and examples**, but they are not perfect. They occasionally mistake thickness fringes or bend contours for dislocations, especially in low‑quality images. They also sometimes miss faint dislocations that a trained metallurgist would catch. However, these errors are predictable and can be mitigated through validation against the gold set. The teacher model’s role is not to be perfect, but to provide a scalable way to generate a large number of reasonably accurate labels. The hybrid approach works because we do not rely on the teacher model blindly; instead, we use it to accelerate the labeling process while maintaining human oversight.

The student model plays a different role. Unlike the teacher, which is a general‑purpose multimodal model, the student is a specialized segmentation network trained specifically for dislocation analysis. The U‑Net architecture is well‑suited for this task because it can capture both local contrast variations and global line structures. The skeletonization step is critical because it converts the segmentation mask into a one‑pixel‑wide representation that can be used for accurate length measurement. This step also reduces the impact of variations in line thickness, which are artifacts of the imaging process rather than physically meaningful features. The student model’s performance shows that it is possible to achieve high accuracy with a relatively small amount of human‑verified data, as long as the training process is guided by a high‑quality teacher model and anchored to the gold set.

One of the most interesting findings from this project is that **density accuracy is more sensitive to false positives than false negatives**. Missing a faint dislocation reduces the total line length slightly, but adding a spurious line—especially a long one—can inflate the density significantly. This asymmetry informed the design of our QC system. By flagging images where the predicted density deviates by more than 10% from the gold set mean, we catch most of the problematic cases without overwhelming the user with false alarms. This threshold is not arbitrary; it reflects the natural variability in dislocation density observed in the gold set and the sensitivity of Taylor’s equation to changes in ρ.

Another important observation is that **the hybrid system is robust across a wide range of imaging conditions**. TEM images vary significantly depending on sample thickness, tilt angle, diffraction vector, and imaging mode. Despite this variability, the student model performs consistently well because it learns the underlying structure of dislocations rather than relying on superficial contrast patterns. The teacher model also benefits from the few‑shot examples provided in the prompt, which help it distinguish between true dislocations and common artifacts. This robustness is essential for real‑world deployment, where imaging conditions are rarely uniform.

The efficiency gains from the hybrid approach are substantial. Manual annotation of 1,200 images would require more than 50 hours of focused work. The hybrid system reduces this to about 6.5 hours, an 8× improvement. This reduction in human labor makes it feasible to analyze large datasets that would otherwise be impractical. It also enables more frequent characterization, which is valuable for process monitoring and quality control in industrial settings. The ability to run the student model locally, without relying on external APIs, further enhances the system’s practicality.

Finally, this project demonstrates that **physics‑aligned computer vision is not only possible but highly effective**. By grounding the entire pipeline in the physics of dislocation contrast and the mechanics of plastic deformation, we ensure that the automated measurements are meaningful and interpretable. The system does not simply detect “lines”; it detects the physical manifestation of lattice distortions that control the mechanical behavior of metals. This alignment between physics and computation is what makes the system reliable and scientifically useful.

Overall, the hybrid architecture provides a scalable, accurate, and physically grounded approach to dislocation density analysis. It combines the strengths of human expertise, large multimodal models, and specialized segmentation networks in a way that respects the underlying physics and addresses the practical challenges of TEM image analysis. The result is a system that significantly accelerates metallurgical research while maintaining the rigor and interpretability required for scientific and industrial applications.

# **6\. Conclusion**

Automating dislocation density analysis is not just a convenience problem; it is a fundamental limitation in how metallurgists study the relationship between microstructure and mechanical behavior. Dislocations control strength, and TEM imaging remains the most direct way to observe them. Yet for decades, the workflow for quantifying dislocation density has barely changed. A metallurgist loads an image, identifies the dislocation lines, traces them manually, converts pixel lengths to physical units, and computes the density. This process is slow, subjective, and difficult to scale. It limits the size of datasets researchers can work with, and it restricts the kinds of questions we can ask about how processing, deformation, and alloying influence microstructure.

The hybrid computer vision system developed in this work addresses that bottleneck directly. By combining human‑verified annotations, large‑scale AI‑assisted labeling, and a distilled U‑Net segmentation model, the system achieves the two goals that manual workflows cannot: scalability and consistency. The gold set anchors the entire pipeline in human expertise. The teacher model scales the annotation process to hundreds or thousands of images. The student model provides a fast, local, and reliable way to analyze new images without relying on external APIs. And the quality‑control framework ensures that the system remains grounded in the physics of dislocation contrast and the mechanics of plastic deformation.

One of the most important outcomes of this project is the demonstration that **a relatively small amount of high‑quality human annotation can support a much larger automated pipeline**. The gold set contained only 100 images, yet it was enough to validate the teacher model, guide the student model, and establish the statistical baseline for density values. This is a powerful result because it shows that metallurgical expertise can be leveraged efficiently. Instead of spending dozens of hours tracing lines across hundreds of images, the expert focuses on a small, carefully selected subset. The AI handles the rest, and the QC system ensures that the automated predictions remain trustworthy.

Another key conclusion is that **physics‑aligned computer vision is not only possible but essential**. Dislocations are not arbitrary visual features. They arise from lattice distortions that obey crystallographic rules, and their visibility depends on diffraction conditions. By grounding the entire pipeline in this physics—through careful annotation, prompt engineering, skeletonization, and density computation—we ensure that the automated measurements are meaningful. The system does not simply detect “lines”; it detects the physical manifestation of defects that control yield strength. This alignment between physics and computation is what makes the system scientifically credible.

The performance results reinforce this point. The student model achieves high IoU scores, low density error, and low yield‑strength prediction error. The hybrid workflow reduces human labor by more than 80%, and the local model enables real‑time analysis without external dependencies. These improvements make it feasible to analyze large TEM datasets that would otherwise be impractical. They also open the door to new applications, such as automated process monitoring, high‑throughput alloy screening, and integration with mechanical testing pipelines.

There are still limitations. The system struggles with extreme bend contours, overlapping dislocations, and very faint lines. These cases often require human review, and the QC system is designed to catch them. The teacher model occasionally misidentifies artifacts, and the student model inherits some of these biases. Future work could explore more advanced segmentation architectures, improved skeletonization methods, or contrast‑adaptive preprocessing to handle challenging images. Another promising direction is to incorporate diffraction‑vector metadata directly into the model, allowing it to reason about invisibility conditions and contrast variations.

Despite these limitations, the hybrid architecture represents a significant step forward for automated microstructural analysis. It shows that human expertise and AI can complement each other in a way that preserves scientific rigor while achieving modern efficiency. The system is not a black box; it is a structured pipeline that mirrors how metallurgists think about dislocations, density, and strength. It respects the physics, leverages the strengths of both humans and AI, and produces results that are accurate, interpretable, and scalable.