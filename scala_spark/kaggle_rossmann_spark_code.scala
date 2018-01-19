// this is a simple: using spark mllib pipeline to solve a regression problem
//

import org.apache.log4j.{Logger}
//core and SparkSQL
import org.apache.log4j.Level
import org.apache.spark.{SparkConf, SparkContext}
// Spark config

import org.apache.spark.sql.hive.HiveContext
// hive tool

import org.apache.spark.sql.SQLContext
//sql tool

import org.apache.spark.sql.DataFrame
// hive DataFrame for storing Data

import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
// ML Feature Creation

import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
//import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
// there is no TrainValidationSplit ??
// tuning hyper parameters

import org.apache.spark.ml.evaluation.{RegressionEvaluator}
// evaluation for regression model

import org.apache.spark.ml.regression.{LinearRegression}
// regression model

import org.apache.spark.ml.Pipeline
// pipeline

import org.apache.spark.mllib.evaluation.RegressionMetrics
// evaluation for regression model


object pipeline {

/*
val stateHolidayIndexer = new StringIndexer()
  .setInputCol("StateHoliday")
  .setOutputCol("StateHolidayIndex")
  */
val schoolHolidayIndexer = new StringIndexer()
  .setInputCol("SchoolHoliday")
  .setOutputCol("SchoolHolidayIndex")

//Convert numerical based categorical features(in one column) to numerical continuous features 
//(one column per category) /increasing sparsity/
  
  /*
val stateHolidayEncoder = new OneHotEncoder()
  .setInputCol("StateHolidayIndex")
  .setOutputCol("StateHolidayVec")
  */
val schoolHolidayEncoder = new OneHotEncoder()
  .setInputCol("SchoolHolidayIndex")
  .setOutputCol("SchoolHolidayVec")
/*
val dayOfMonthEncoder = new OneHotEncoder()
  .setInputCol("DayOfMonth")
.setOutputCol("DayOfMonthVec")
*/
val dayOfWeekEncoder = new OneHotEncoder()
  .setInputCol("DayOfWeek")
.setOutputCol("DayOfWeekVec")
/*
val storeEncoder = new OneHotEncoder()
  .setInputCol("Store")
  .setOutputCol("StoreVec")
*/

// all the features would transformed to one-hot features

//assemble all of our vectors together into one vector to input into our model.
val assembler = new VectorAssembler()
  .setInputCols(Array(
              //"StoreVec", 
              "DayOfWeekVec", 
              "Open",
              //"DayOfMonthVec", 
              //"StateHolidayVec", 
              "SchoolHolidayVec"
            )
        )
  .setOutputCol("features")



//------------------------------------ Creating the Pipelines -------------------------------------
/*Much like the DAG that we saw in the the core concepts of the Pipeline:
Transformer: 
A Transformer is an algorithm which can transform one DataFrame into another DataFrame.
E.g., an ML model is a Transformer which transforms DataFrame with features into a DataFrame with predictions.
Estimator: 
An Estimator is an algorithm which can be fit on a DataFrame to produce a Transformer. 
E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model.
Pipeline: 
A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow.
*/

/*
Once you see the code, you'll notice that in our train test split we're also setting an Evaluatorthat 
will judge how well our model is doing and automatically select the best parameter for us to use based 
on that metric. This means that we get to train lots and lots of different models to see which one is best. 
Super simple! Let's walk through the creation for each model.
*/

// this is a pipeline which can cerate new features and train the model and tune the parameters
// finally , this pipeline would give us a model with best parameters 
def preppedLRPipeline():CrossValidator = {
  val lr = new LinearRegression()

  val paramGrid = new ParamGridBuilder()  // parameters to tune
    .addGrid(lr.regParam, Array(0.1, 0.01))
    //.addGrid(lr.fitIntercept)
    .addGrid(lr.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
    .build()

  val pipeline = new Pipeline()
    .setStages(Array(                   // put all the preprocessing before and model into the pipeline
            //stateHolidayIndexer, 
            schoolHolidayIndexer,
              //stateHolidayEncoder, 
              schoolHolidayEncoder, 
              //storeEncoder,
              dayOfWeekEncoder, 
              //dayOfMonthEncoder,
              assembler, 
              lr  // the last thing is the model
              )
        )

  val tvs = new CrossValidator()  // put the [preprocessing, model], Evaluator, tuning, paramgrid together
    .setEstimator(pipeline) // here is something can be fit
    .setEvaluator(new RegressionEvaluator) //here is a model Evaluator
    .setEstimatorParamMaps(paramGrid) // here is the hyper parameters to tune
    .setNumFolds(4) 

  tvs  // return the model with best parameters
}


// -------------------- save predictions --------------------------------------
def savePredictions(predictions:DataFrame, testRaw:DataFrame) = {
  val tdOut = testRaw
    .select("Id") // select the colunms of "Id"
    //.distinct()
    .join(predictions, testRaw("Id") === predictions("PredId"), "outer") // combine acorrding to id
    .select("Id", "Sales")
    .na.fill(0:Double) // some of our inputs were null so we have to
                       // fill these with something
  tdOut.rdd.saveAsTextFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/linear_regression_predictions_02.csv")
  tdOut.rdd.foreach(x=>println(x))
  
}


// ------Fitting, Testing, and Using The Model-------------------------
/*
Now we've brought in our data, created our pipeline, 
we are now ready to train up our models and see how they perform. 
This will take some time to run because we are exploring a 
hyperparameter space for each model. It takes time to try out all 
the permutations in our parameter grid as well as create a training 
set for each tree so be patient!
*/
def fitModel(tvs:CrossValidator, data:DataFrame) = {
  val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 12345)
  // get the train data and test data

  println("Fitting data")
  val model = tvs.fit(training) // fit the model using tvs(a pipeline)(including tunning hyperparameters !)

  println("Now performing test on hold out set")
  val holdout = model.transform(test).select("prediction","label") // result of test

  // have to do a type conversion for RegressionMetrics
  val rm = new RegressionMetrics(holdout.rdd.map(x =>
    (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

  println("Test Metrics")
  println("Test Explained Variance:")
  println(rm.explainedVariance)
  println("Test R^2 Coef:")
  println(rm.r2)
  println("Test MSE:")
  println(rm.meanSquaredError)
  println("Test RMSE:")
  println(rm.rootMeanSquaredError)

  model //return the model with best parameters 
}

def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }

//###############################################################################################
def main(args: Array[String]): Unit = {

  SetLogger()
  
  
//val conf = new SparkConf().setAppName("pipeline_regression")
val sc = new SparkContext(new SparkConf().setAppName("App").setMaster("local[4]"))
val sqlContext = new SQLContext(sc)

//============== reading data ==========================

// reading train data----------------------
println("begin read train data !")
val dt1 = sc.textFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/train.csv", 1)
val trainRaw = dt1.map(line => line.split(","))
val train = trainRaw.map(x => R_train(
										x(0).toInt,
										x(1).toInt,
										x(2).toString,
										x(3).toDouble,
										x(4).toInt,
										x(5).toInt,
										x(6).toInt,
										//x(7).toString,
										x(8).toString
										)
)
//val trainRaw = dt1.map(s => Vectors.dense(s.split(',').map(_.toDouble)))
println("finish reading train data!")


// reading test data--------------------------------
println("begin to read test data !")
val dt2= sc.textFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/test.csv", 1)
val testRaw = dt2.map(line => line.split(","))
val test = testRaw.map(x => R_train(
										x(0).toInt,
										x(1).toInt,
										x(2).toString,
										x(3).toDouble,
										x(4).toInt,
										x(5).toInt,
										x(6).toInt,
										//x(7).toString,
										x(8).toString
										)
)
println("finish  reading test data !")


//reading store data -------------------------
println("begin to read store data")
val dt3= sc.textFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/store.csv", 1)
val storeRaw = dt3.map(line => line.split(","))
val store = storeRaw.map(x => store_info(
										x(0).toInt,
										x(1).toString,
										x(2).toString,
										x(3).toInt,
										x(4).toInt,
										x(5).toInt,
										x(6).toInt,
										x(7).toInt,
										x(8).toInt
							
))
println("finish reading store data !")


// transfrom RDD into dataFrame--------------------
import sqlContext.implicits._

val train_data = train.toDF()
val test_data = test.toDF()
val store_data = store.toDF()

train_data.registerTempTable("table_train")
test_data.registerTempTable("table_test")
store_data.registerTempTable("store_table")

// using sql to get the data we want from table-------
val train_data_sql = sqlContext.sql("""
			SELECT 
                Store,
                Sales label, 
                Open Open, 
                DayOfWeek DayOfWeek,
                SchoolHoliday
            FROM table_train
            Limit 100
          """).na.drop()
train_data_sql.show(10)

val test_data_sql = sqlContext.sql("""
			SELECT 
                Store,
                Sales label, 
                Open Open, 
                DayOfWeek DayOfWeek,
                SchoolHoliday
            FROM table_test
          """).na.drop()
test_data_sql.show(20)

val store_data_sql = sqlContext.sql("""
			SELECT *
            FROM store_table
          """).na.drop()

store_data_sql.show(20)

//============== try to combine train data and store data together ==========
store_data_sql.registerTempTable("s")
train_data_sql.registerTempTable("t")
val store_train_data_sql = sqlContext.sql("""
			SELECT s.Store, t.Store
            FROM s ,   t
            WHERE s.Store = t.Store
          """).na.drop()

store_train_data_sql.show(20)



/*


// ================== main step ======================

val linearTvs = preppedLRPipeline()
// The linear Regression Pipeline
println("finish building  pipleine")


println("evaluating linear regression")
val lrModel = fitModel(linearTvs, train_data_sql)
println("finish training fitting !")

println("Generating kaggle predictions")
val lrOut = lrModel.transform(train_data_sql) // transform is predicting in the sklearn
  .withColumnRenamed("prediction","Sales") //rename the column

lrOut.rdd.foreach(row => println(row))
println("Saving kaggle predictions")

//savePredictions(lrOut, test_data_sql)
lrOut.rdd.saveAsTextFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/linear_regression_predictions_03.csv")
println("saveing is done !")

*/

}



}

case class R_train(
Store:Int,
DayOfWeek:Double,
Date:String,
Sales:Double,
Customers:Int,
Open:Int,
Promo:Int,
//StateHoliday:String,
SchoolHoliday:String
)

case class store_info(
Store:Int,
StoreType:String,	
Assortment:String,
CompetitionDistance: Int,
CompetitionOpenSinceMonth:Int,	
CompetitionOpenSinceYear:Int,
Promo2:Int,
Promo2SinceWeek:Int,
Promo2SinceYear:Int
)
