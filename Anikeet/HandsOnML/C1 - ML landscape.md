## What is Machine learning
- Machine learning is science or art which give machine power to learn without being explicitly programmed

## What are different types of Machine learning algorithms
1. How machines are supervised (supervised vs unsupervised)
2. Can the machines learn incrementally ? (online vs batch)
3. value access vs predict (instance based vs model-based)
### How machines are supervised
##### Supervised machine learning
	Algorithms in which we feed data and desired solution

##### Unsupervised machine learning
	Algorithms in which we do not provide labeled data for traning a ML model
			Some types are 
				- Data visualization - visual 2d or 3d graphs
				- Anomly detection - detects somethings that is not usual
				- Associate Rule learning - Detect relations between diff objects
##### Semi supervised ML
		Algorithm clusters similar objects togeahter and then they are labeled manually, this data is can now be labeled data

##### Self supervised ML
	Same as semi supervised ML but we just dont assign labels we work with cluseters

##### Reinforcement Learning
	By giving a punishment or reward for action it performs

### Can the Machines learn incrementally?

##### Batch Learning
	ML algorithms learn on a complete set on data, the algorithms might be retarined 

##### Online Learning
	ML algorithms are fed data in small pieces and then later incrementally to keep model uptodate with latest infomation


### Value access vs predict
##### Instance Based learning
	Similarity based prediction

##### Model Based learning
	Learn from a series of values and predic numbers

### Main challenges in ML
- Insufficient quantity of data
- Non Representative data
- Poor quality data
- Irrelevant features
- Over-fitting data
- Under-fitting data

### Testing and Validation
- Data can overfit or under fit to make sure our model generalises well to new data we need to check (Test and validate) our data

#### Hyperpatameter Tuning and model selection

- We can try and choose different models and for each model we can choose different parameters to obtain best model. We can train multiple models on various parameters and choose best one
- A common solution to train avoid over-fitting and training multiple times is using a cross validation (cv) set, a when after its trained is tested on a cv set and if doesn't perform well on cv set the model is over-fitting else model is ready
#### Data mismatch
- Sometime, We don't have enough data to train on, we use e.g. we have 3000 photos of cats and dogs taken from a phone but its not enough so we take 10000 photos from web, here we need to make sure we put a large amount of photos in cv and test set







