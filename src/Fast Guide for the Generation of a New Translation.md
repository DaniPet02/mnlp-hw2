The following guide explains how to obtain a translation of a set of sentences written in archaic Italian into their corresponding modern form, using a selection of proposed models.

### Requested Characteristics for the Execution Environment
The execution environment (Colab or local) must support GPU acceleration with an Nvidia GPU providing at least 12GB of dedicated memory, in order to use the mentioned quantization methods (QLoRA, mixed-precision floating point formats such as bfloat, etc...).

### Generation Pipeline

The generation pipeline involves modifying and running the two notebooks `prompting.ipynb` and `evaluation.ipynb` located in the project folder, sequentially as described below:

1. Upload the dataset to be processed into the root directory of the project
2. Update the `DATASET_EVAL` variable in the Globals section of the `prompting.ipynb` file
3. Select a model from the available ones, paired with a compatible prompt (see description in the notebook)
4. Run the entire notebook. Upon completion, a file in *{CSV/JSONL/XCEL}* format will be generated in the `unnot/` folder, containing the model prompt, the original sentence, and the corresponding model-generated translation

Once tranlsation file is generated, it can be:
Una volta generato il file con le traduzioni questo può essere:

* further manually annotated using the other models included in the experiment (ChatGPT and Gemini)
* used to proceed with the validation process via Prometheus-eval as the LLM Judge (local execution)


The evaluation phase of the generated output is performed simply by running the entire `evaluation.ipynb` notebook. The code within the notebook will automatically carry out the following steps:
1. Evaluate all files in the `unnot/` folder with extension and format compliant with .excel or similar formats
2. Generate a final report folder based on the name assigned to the evaluated file, which must follow the naming convention for optimal results:        * \[model_name\]\(prompt_type\)
3. The generated files will be in JSONL format only, as requested

For the Fine-Tuning process, reference is made directly to the `fine-tuning.ipynb` notebook, structured similarly to the two previously presented notebooks and thoroughly annotated.


#### Guide made by:
*Many Naps Little Progress* 
