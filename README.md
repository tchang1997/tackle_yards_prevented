# TAPER: An Intuitive Metric for Measuring Yards Prevented by Tackles 

Official code repository for 2024 NFL Big Data Bowl (Kaggle) Metrics track submission.**

![Example play with model predictions](./gifs/deebo-final.gif)


# Pipeline

To train your own models on the tracking data:
1. Run `preprocess_data.py` to generate the geometric and raw tracking features needed.
2. Create a configuration file in `configs/`.
3. Run `python main.py --config [YOUR_CONFIG]`. You can optionally visualize model training and validation statistics via `tensorboard --logdir .`.
4. Evaluate models using `python evaluate.py --ckpt-dir [YOUR_MODEL_DIR]`.
5. Use `postprocessed_eda.ipynb` to further analyze model outputs, and create interactive animations for each play to visualize model predictions. 
6. Optionally run `python animate.py --ckpt-path [YOUR_MODEL_DIR] --split [train,val,test] --indices [LIST_OF_INDICES]` to create an interactive HTML play visualization with model predictions. 
