using Microsoft.ML;
using Microsoft.ML.Data;

var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var assetsRelativePath = Path.Combine(projectDirectory, "assets");
var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");

Console.WriteLine("Press A to train and save a model, press B to make predictions with a saved model");

var answer = Console.ReadKey();

switch (answer.KeyChar.ToString().ToUpper())
{
	case "A":
		TrainModel();
		break;
	case "B":
		MakePredictions();
		break;
	default:
		Console.WriteLine("Wrong answer!");
		break;

}

void TrainModel()
{
	MLContext mlContext = new MLContext();

	IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, "Life Expectancy Data.csv"), hasHeader: true, separatorChar: ',');

	// Exempel på pipeline som använder några features från csv-filen för att förutsäga life expectancy:
	// Du kan använda denna eller skapa en egen, t.ex. via Model Builder
	// Du kan variera vilka features (kolumner i csv-filen) du vill inkludera i din modell
	var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "StatusEncoded", inputColumnName: "Status")
		.Append(mlContext.Transforms.Concatenate("Features", "StatusEncoded", "AdultMortality", "InfantDeaths", "Alcohol", "PercentageExpenditure", "HepatitisB", "Measles", "BMI"))
		.Append(mlContext.Regression.Trainers.FastTree());

	// Dela upp i test- och träningsdata
	// Ledtråd: använd metoden TrainTestSplit i MLContext
	IDataView trainSet = null;
	IDataView testSet = null;

	// Träna modellen på träningsdata
	// Ledtråd: använd metoden Fit i pipeline, lägg in i variabeln trainedModel
	ITransformer trainedModel = null;

	// Spara modellen i workspace-mappen
	SaveModel(mlContext, trainedModel, dataView);
	
	// Utvärdera modellen
	Evaluate(mlContext, trainedModel, testSet);
}

void SaveModel(MLContext mlContext, ITransformer trainedModel, IDataView data)
{
	// Save Trained Model
	mlContext.Model.Save(trainedModel, data.Schema, Path.Combine(workspaceRelativePath, "model.zip"));
}

void Evaluate(MLContext mlContext, ITransformer model, IDataView testData)
{
	var predictions = model.Transform(testData);
	var metrics = mlContext.Regression.Evaluate(predictions);

	// Printa ut några metrics till consolen!
	Console.WriteLine();
	Console.WriteLine($"*************************************************");
	Console.WriteLine($"*       Model quality metrics evaluation         ");
	Console.WriteLine($"*------------------------------------------------");

}

void MakePredictions()
{
	MLContext mlContext = new MLContext();

	// Ladda in modellen
	ITransformer model = mlContext.Model.Load(Path.Combine(workspaceRelativePath, "model.zip"), out _);

	// Skapa en prediction engine
	var predictionFunction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

	// Gör förutsägelser
	// Skapa en input och förutsäg life expectancy för den
	var sample = new ModelInput()
	{ Status = "Developing", AdultMortality = 263, InfantDeaths = 62, Alcohol = 0.01f, PercentageExpenditure = 71.27962362f, HepatitisB = 65, Measles = 1154, BMI = 19.1f, LifeExpectancy = 65 };
	
	var prediction = predictionFunction.Predict(sample);

	// Skriv ut vad det förutsagda värdet blev, samt värdet i det faktiska exempelet (sample). Jämför.
}

public class ModelInput
{
	[LoadColumn(2)]
	public string? Status;

	[LoadColumn(3), ColumnName("Label")]
	public float LifeExpectancy;

	[LoadColumn(4)]
	public float AdultMortality;

	[LoadColumn(5)]
	public float InfantDeaths;

	[LoadColumn(6)]
	public float Alcohol;

	[LoadColumn(7)]
	public float PercentageExpenditure;

	[LoadColumn(8)]
	public float HepatitisB;

	[LoadColumn(9)]
	public float Measles;

	[LoadColumn(10)]
	public float BMI;

}

public class ModelOutput
{
	[ColumnName("Score")]
	public float LifeExpectancy;
}