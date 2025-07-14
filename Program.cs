using System;
using Microsoft.ML;
using Microsoft.ML.Data;

// Step 1: Define data classes
public class SalesData
{
    [LoadColumn(0)]
    public float Month;

    [LoadColumn(1)]
    public float Sales;
}

public class SalesPrediction
{
    [ColumnName("Score")]
    public float ForecastedSales;
}

class Program
{
    static void Main(string[] args)
    {
        // Step 2: Create MLContext
        var mlContext = new MLContext();

        // Step 3: Load Data
        var dataPath = "sales-data.csv";
        var dataView = mlContext.Data.LoadFromTextFile<SalesData>(dataPath, hasHeader: true, separatorChar: ',');

        // Step 4: Create pipeline
        var pipeline = mlContext.Transforms.Concatenate("Features", nameof(SalesData.Month))
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Sales", maximumNumberOfIterations: 100));

        // Step 5: Train model
        var model = pipeline.Fit(dataView);

        // Step 6: Create prediction engine
        var predictor = mlContext.Model.CreatePredictionEngine<SalesData, SalesPrediction>(model);

        // Step 7: Forecast
        for (int month = 7; month <= 12; month++)
        {
            var input = new SalesData { Month = month };
            var prediction = predictor.Predict(input);
            Console.WriteLine($"Month: {month} => Predicted Sales: {prediction.ForecastedSales:F2}");
        }
    }
}
