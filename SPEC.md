# Waste Image–Caption Dataset Curation Project

## Problem Statement
Waste images often contain multiple, overlapping, damaged, or partially occluded objects such as plastic bottles, metal cans, wrappers, cartons, and food containers. Although general vision-language models such as BLIP, LLaVA, and Qwen2.5-VL can generate image captions, their descriptions may be incomplete or inaccurate for waste scenes and usually do not consistently capture material, object condition, and clutter context. Publicly available waste datasets provide images and detection/segmentation labels, but there is limited availability of carefully curated image–caption datasets tailored to waste understanding. 

This project aims to develop a verified waste image–caption dataset by generating captions with pretrained vision-language models and refining them through structured quality checks and human review. The final dataset will describe waste category, material, physical state, and scene context, providing a useful resource for future waste-focused VLM and VLA research.

## Objectives
* **Data Aggregation**: Collect a large pool of waste images from open sources (TACO, ZeroWaste, COCO classes).
* **Caption Generation**: Use pretrained VLMs (BLIP, LLaVA, Qwen-VL) to auto-generate captions for waste images.
* **Quality Control**: Develop metrics and processes (e.g., CLIPScore, human checks) to filter and improve generated captions.
* **Structured Language**: Encourage structured captions including object, material, state (e.g., “crushed plastic bottle”) to align with waste taxonomy.

## Tasks

### Week 1 – Project Setup and Dataset Preparation
Study TACO and ZeroWaste datasets; finalize waste categories, annotation format, and project scope. Collect, clean, and organize diverse waste images into training, validation, and test splits. Define the annotation schema, including object category, material, condition, location, and sorting action.

### Week 2 – Caption Generation
Implement a pretrained vision-language model (VLM) to generate object-level captions describing waste objects, their materials, conditions, and spatial relationships. Generate initial captions for the selected images and associate them with available bounding boxes or segmentation masks.

### Week 3 – Annotation Verification and Instruction Generation
Manually verify and correct generated captions, removing hallucinated or unsupported descriptions. Create grounded natural-language sorting instructions and link each instruction to its corresponding object bounding box or segmentation mask.

### Week 4 – Dataset Finalization and Baseline Evaluation
Organize the verified image–caption–instruction pairs into a structured JSON dataset. Validate annotation consistency and evaluate a baseline VLM for caption generation and object grounding. Record caption quality, hallucination rates, and grounding accuracy.

### Week 5 – Final Evaluation and Documentation
Perform error analysis, finalize train/validation/test splits, and prepare the dataset card, annotation guidelines, GitHub repository, evaluation results, final report, and demonstration video.

## Datasets & Tools
* **Datasets**: TACO waste dataset, ZeroWaste, COCO (relevant classes like “bottle”, "can"), in-the-wild images from search.
* **Models**: BLIP/BART, LLaVA (pretrained), Qwen-VL (for variation).
* **Tools**: Python, PyTorch / Hugging Face; evaluation metrics (CIDEr, BLEU from MS COCO toolkit); CLIPScore.

## Expected Deliverables
**Streamlit-based Waste Caption Dataset Curation App** for generating, reviewing, editing, filtering, and exporting waste-image captions. 

### Required Features:
* Upload/select waste images and display image metadata.
* Select a captioning model and enter a prompt template.
* Generate one or multiple candidate captions.
* Edit, approve, reject, or regenerate captions manually.
* Add structured fields: object, material, condition, count, and scene context.
* Show quality score/review score and filtering options.
* Display side-by-side outputs from BLIP-2 and Qwen2.5-VL.
* Export approved annotations as JSON/JSONL and CSV.
* Show basic dataset statistics: number of images, class distribution, approval rate, and captions per class.

## References

1. **TACO Dataset**  
   Proença, P. F., & Simões, P. (2020). TACO: Trash Annotations in Context for Litter Detection. *arXiv preprint arXiv:2003.06975*.  
   [Paper (arXiv)](https://arxiv.org/abs/2003.06975) | DOI: [10.48550/arXiv.2003.06975](https://doi.org/10.48550/arXiv.2003.06975)

2. **ZeroWaste Dataset**  
   Bashkirova, D., Abdelfattah, M., Zhu, Z., Akl, J., Alladkani, F., Hu, P., Ablavsky, V., Calli, B., Bargal, S. A., & Saenko, K. (2022). ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 21147–21157.  
   [Paper (CVF)](https://openaccess.thecvf.com/content/CVPR2022/html/Bashkirova_ZeroWaste_Dataset_Towards_Deformable_Object_Segmentation_in_Cluttered_Scenes_CVPR_2022_paper.html) | DOI: [10.48550/arXiv.2106.02740](https://doi.org/10.48550/arXiv.2106.02740)

3. **BLIP**  
   Li, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. *International Conference on Machine Learning (ICML)*, 12888–12900.  
   [Paper (arXiv)](https://arxiv.org/abs/2201.12086) | DOI: [10.48550/arXiv.2201.12086](https://doi.org/10.48550/arXiv.2201.12086)

4. **LLaVA (Visual Instruction Tuning)**  
   Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning. *Advances in Neural Information Processing Systems (NeurIPS)*, 36, 34892–35016.  
   [Paper (arXiv)](https://arxiv.org/abs/2304.08485) | DOI: [10.48550/arXiv.2304.08485](https://doi.org/10.48550/arXiv.2304.08485)

5. **Qwen2.5-VL**  
   Qwen Team. (2025). Qwen2.5-VL Technical Report. *arXiv preprint arXiv:2502.13923*.  
   [Paper (arXiv)](https://arxiv.org/abs/2502.13923) | DOI: [10.48550/arXiv.2502.13923](https://doi.org/10.48550/arXiv.2502.13923)

6. **CLIPScore**  
   Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., & Choi, Y. (2021). CLIPScore: A Reference-free Evaluation Metric for Image Captioning. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 7551–7561.  
   [Paper (ACL Anthology)](https://aclanthology.org/2021.emnlp-main.595.pdf) | DOI: [10.18653/v1/2021.emnlp-main.595](https://doi.org/10.18653/v1/2021.emnlp-main.595)

7. **MS COCO Dataset**  
   Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. *European Conference on Computer Vision (ECCV)*, 740–755.  
   [Paper (arXiv)](https://arxiv.org/abs/1405.0312) | DOI: [10.48550/arXiv.1405.0312](https://doi.org/10.48550/arXiv.1405.0312)

8. **BLIP-2**  
   Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. *International Conference on Machine Learning (ICML)*, 19730–19742.  
   [Paper (arXiv)](https://arxiv.org/abs/2301.12597) | DOI: [10.48550/arXiv.2301.12597](https://doi.org/10.48550/arXiv.2301.12597)

9. **BART Model**  
   Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 7871–7880.  
   [Paper (ACL Anthology)](https://aclanthology.org/2020.acl-main.703.pdf) | DOI: [10.18653/v1/2020.acl-main.703](https://doi.org/10.18653/v1/2020.acl-main.703)

10. **CIDEr Evaluation Metric**  
    Vedantam, R., Zitnick, C. L., & Parikh, D. (2015). CIDEr: Consensus-based Image Description Evaluation. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4566–4575.  
    [Paper (CVF)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf) | DOI: [10.1109/CVPR.2015.7299087](https://doi.org/10.1109/CVPR.2015.7299087)

11. **BLEU Evaluation Metric**  
    Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL)*, 311–318.  
    [Paper (ACL Anthology)](https://aclanthology.org/P02-1040.pdf) | DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
