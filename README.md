# MLE Home Assignment

## Getting Started

To run the following code you can use one of two ways:
1. Using Docker
2. Run locally

To run the code the first step will be to clone the repo. Use: 
```
git clone https://github.com/saharBaribi/MLEHomeAssignment.git
cd MLEHomeAssignment
```

### Run Using Docker 
If you don't have Docker Desktop, you first need to download it. 
Build the Docker image: 
```
docker build -t riskified-mle .
```

Run the Docker container: 
```
docker run riskified-mle
```


### Run Locally 
First you should install the required packages. Run the following line: 

``` 
pip install -r requirements.txt
```

Then, run the application: 

```
python main.py
```

## Testing
There are two types of tests in the current application: 
1. Unittesting - can be either run by pytest or just running the relevant function in the following path:
   ```
   python src/Testing/unit_testing.py
   ```
2. Data validation during runtime - will happen while you run the main function. 




 
