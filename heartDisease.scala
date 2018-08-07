// Written by: Nicholas Cockcroft
// Date: August 2, 2018
// Assignment: Assignment #8

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.rdd._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}


// Open file and store the data after the first column into an rdd vector except for the last column
val file = sc.textFile("/home/nick/input/hearts.txt")
val splitData = file.map(x=> x.split(","))
// Also, many of the fields can have a "?" in it so I am checking for those and replacing them with a -1 if they are in the field
val rawData = splitData.map(x=>(x(0).toDouble,x(1).toDouble, x(2).toDouble, 
		if(x(3) == "?") {-1.toDouble} else {x(3).toDouble}, 
		if(x(4) == "?") {-1.toDouble} else {x(4).toDouble}, 
		if(x(5) == "?") {-1.toDouble} else {x(5).toDouble}, 
		if(x(6) == "?") {-1.toDouble} else {x(6).toDouble}, 
		if(x(7) == "?") {-1.toDouble} else {x(7).toDouble}, 
		if(x(8) == "?") {-1.toDouble} else {x(8).toDouble}, 
		x(9).toDouble, 
		if(x(10) == "?") {-1.toDouble} else {x(10).toDouble}, 
		if(x(11) == "?") {-1.toDouble} else {x(11).toDouble}, 
		if(x(12) == "?") {-1.toDouble} else {x(12).toDouble}))

// Creating a vector with all of the date fields to be fed into the kmenas clusters and make a prediction
val parsedData = rawData.map(x=> Vectors.dense(x._2,x._3,x._4,x._5,x._6,x._7,x._8,x._9,x._10,x._11,x._12,x._13))

// Training the cluster with the vector data
val kmeans = new KMeans()
kmeans.setK(2)
val model = kmeans.run(parsedData)

// Now obtaining each persons individual prediction and mapping the id as the key and the prediction as the value
val cluster = rawData.map(x=>(x._1.toDouble,model.predict(Vectors.dense(x._2,x._3,x._4,x._5,x._6,x._7,x._8,x._9,x._10,x._11,x._12,x._13))))

// Outputting the predictions with the person's id and the prediction
cluster.collect.foreach(println)
