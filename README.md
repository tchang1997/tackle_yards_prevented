# Did He Really Save that Touchdown? An Intuitive Metric for Measuring Yards Prevented by Tackles

Official code repository for 2024 NFL Big Data Bowl (Kaggle) submission **"Did He Really Save that Touchdown? An Intuitive Metric for Measuring Yards Prevented by Tackles."**


# Demo

**TODO: Large GIF goes here with a play description.** 

# Pipeline

To train your own models on the tracking data:
1. Run `preprocess_data.py` to generate the geometric and raw tracking features needed.
2. Create a configuration file in `configs/`.
3. Run `python main.py --config [YOUR_CONFIG]`. You can optionally visualize model training and validation statistics via `tensorboard --logdir .`.
4. Evaluate models using `python evaluate.py --ckpt-dir [YOUR_MODEL_DIR]`.
5. Use `postprocessed_eda.ipynb` to further analyze model outputs, and create interactive animations for each play to visualize model predictions. 
