# ANN-fashion-recommendation

## Medium Article

If you want to read more about this project, check out the following [Medium article](https://medium.com/@summit100/artificial-neural-networks-final-project-fashion-recommendation-6b834041c4eb)

## Summary

This GitHub repo implements a fashion recommnedation system. To summarize, we created a CNN model to categorize clothing types, used the model for feature extraction and created a large embedding space of a diverse set of clothing, then use KNN for an inputted clothing image on this embedding space to make fashion recommendations.  

## File Layout

- **Application.py**: Implements a Streamlab app for users to uploads their images, then get a top 5 recommendations from our system.

- **create-dataset.py**: Selects at most 500 clothing from each category, then reformats the original DeepFashion dataset into three files: test_new.csv, train_new.csv, val_new.csv.

* **embeddings.ipynb**: Creates the embedding space used for recommending clothing. Also sets up qualitative + quantitative tests for the quality of
  recommendations.

- **model.ipynb**: Sets up, trains, and tests the clothing categorization CNN model. The weights of this model can be found in
  model-data/model.h5

## Blog Figures

For our article, we used various stiches of images within our dataset. We created these images using the
PIL library. This code can be found in the embeddings.ipynb file.

## Sources

- **Dataset:** [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- We also used this [GitHub repo](https://github.com/imshreyshah/Clothing-Category-Prediction-DeepFashion) for help in creating our create-dataset.py file.
