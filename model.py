import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Check if dataset files exist
if not os.path.exists('./data/train.csv') or not os.path.exists('./data/test.csv'):
    print("Dataset files are missing! Please download the datasets from Kaggle.")
    exit()

# Print versions of TensorFlow and TensorFlow Decision Forests
print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

# Load and inspect the dataset
dataset_df = pd.read_csv('./data/train.csv')
print("Full train dataset shape is {}".format(dataset_df.shape))
print(dataset_df.head(5))
print(dataset_df.describe())
print(dataset_df.info())

# Visualize the distribution of the target variable
plot_df = dataset_df.Transported.value_counts()
plot_df.plot(kind="bar", title="Transported Value Counts")
plt.show()

# Plot distributions for numerical columns
fig, ax = plt.subplots(5, 1, figsize=(10, 10))
plt.subplots_adjust(top=2)
sns.histplot(dataset_df['Age'], color='b', bins=50, ax=ax[0])
sns.histplot(dataset_df['FoodCourt'], color='b', bins=50, ax=ax[1])
sns.histplot(dataset_df['ShoppingMall'], color='b', bins=50, ax=ax[2])
sns.histplot(dataset_df['Spa'], color='b', bins=50, ax=ax[3])
sns.histplot(dataset_df['VRDeck'], color='b', bins=50, ax=ax[4])
plt.show()

# Drop unnecessary columns and handle missing values
dataset_df = dataset_df.drop(['PassengerId', 'Name'], axis=1)
dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)

# Convert boolean labels and other boolean fields to integers
label = "Transported"
dataset_df[label] = dataset_df[label].astype(int)
dataset_df['VIP'] = dataset_df['VIP'].astype(int)
dataset_df['CryoSleep'] = dataset_df['CryoSleep'].astype(int)

# Split "Cabin" into "Deck", "Cabin_num", and "Side" and drop "Cabin" column
dataset_df[["Deck", "Cabin_num", "Side"]] = dataset_df["Cabin"].str.split("/", expand=True)
dataset_df = dataset_df.drop('Cabin', axis=1)

# Split the dataset into training and validation sets
def split_dataset(dataset, test_ratio=0.20):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print(f"{len(train_ds_pd)} examples in training, {len(valid_ds_pd)} examples in testing.")

# Convert to TensorFlow Dataset format for TFDF compatibility
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label)

# Initialize and compile a TFDF model, for example, using Gradient Boosted Trees for improved accuracy
model = tfdf.keras.GradientBoostedTreesModel()
model.compile(metrics=["accuracy"])

# Train the model
model.fit(x=train_ds)

# Visualize one of the decision trees
tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)

# Out-of-bag evaluation and validation dataset evaluation
logs = model.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.show()

# Get evaluation metrics on validation data
evaluation = model.evaluate(x=valid_ds, return_dict=True)
for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

# Display variable importances
inspector = model.make_inspector()
print("Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("	", importance)
print(inspector.variable_importances()["NUM_AS_ROOT"])

# Load the test dataset
test_df = pd.read_csv('./data/test.csv')
submission_id = test_df.PassengerId

# Handle missing values and convert boolean fields to integers
test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
test_df = test_df.drop('Cabin', axis=1)
test_df['VIP'] = test_df['VIP'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

# Convert to TensorFlow Dataset and make predictions
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)
predictions = model.predict(test_ds)
n_predictions = (predictions > 0.5).astype(bool).squeeze()

# Create submission DataFrame and save to CSV
output = pd.DataFrame({'PassengerId': submission_id, 'Transported': n_predictions})
sample_submission_df = pd.read_csv('./data/sample_submission.csv')
sample_submission_df['Transported'] = n_predictions
sample_submission_df.to_csv('./submission.csv', index=False)

print("Submission file created successfully.")
