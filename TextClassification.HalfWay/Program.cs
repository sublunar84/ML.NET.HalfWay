using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

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
	var mlContext = new MLContext();

	var dataView = mlContext.Data.LoadFromTextFile<ModelInput>(Path.Combine(assetsRelativePath, "Luxury_Products_Apparel_Data.csv"), hasHeader: true, separatorChar: ',');

	// Exempel på pipeline som använder ProductName och Description från csv-filen för att förutsäga Category (jag använde Model Builder för att hitta denna):
	// Du kan använda denna eller skapa en egen, t.ex. via Model Builder eller exempelkod
	// Du kan variera vilka features (kolumner i csv-filen) du vill inkludera i din modell. Du kan även testa att förutsäga SubCategorý istället
	var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName: @"ProductName", outputColumnName: @"ProductName")
		.Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: @"Description", outputColumnName: @"Description"))
		.Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"ProductName", @"Description" }))
		.Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Category", inputColumnName: @"Category", addKeyValueAnnotationsAsText: false))
		.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator: mlContext.BinaryClassification.Trainers.FastForest(
			new FastForestBinaryTrainer.Options() { NumberOfTrees = 4, NumberOfLeaves = 4, FeatureFraction = 1F, LabelColumnName = @"Category", FeatureColumnName = @"Features" }),
			labelColumnName: @"Category"))
		.Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

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

void MakePredictions(IDataView? testSet = null)
{
	MLContext mlContext = new MLContext();

	// Ladda in modellen
	ITransformer model = mlContext.Model.Load(Path.Combine(workspaceRelativePath, "model.zip"), out _);

	// Skapa en prediction engine
	var predictionFunction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

	// Gör förutsägelser
	// Skapa en input och förutsäg life expectancy för den
	var sample = new ModelInput()
	{
		Category = "Accessories",
		ProductName = "Prada Striped Shell Belt Bag",
		Description =
			"One of Prada's most functional designs, this belt bag is made from weather-resistant shell fabric with zip compartments for storing your daily belongings. It's designed for navigating your day hands-free- try styling yours diagonally across the body."
	};

	var prediction = predictionFunction.Predict(sample);

	// Skriv ut vad det förutsagda värdet blev, samt värdet i det faktiska exempelet (sample). Jämför.

}

void Evaluate(MLContext mlContext, ITransformer model, IDataView testData)
{
	var predictions = model.Transform(testData);
	var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Category");

	// Printa ut några metrics till consolen!
	Console.WriteLine();
	Console.WriteLine($"*************************************************");
	Console.WriteLine($"*       Model quality metrics evaluation         ");
	Console.WriteLine($"*------------------------------------------------");

}

public class ModelInput
{
	[LoadColumn(1)]
	[ColumnName(@"Category")]
	public string Category { get; set; }

	[LoadColumn(3)]
	[ColumnName(@"ProductName")]
	public string ProductName { get; set; }

	[LoadColumn(4)]
	[ColumnName(@"Description")]
	public string Description { get; set; }

}

public class ModelOutput
{
	[ColumnName(@"Category")]
	public uint Category { get; set; }

	[ColumnName(@"ProductName")]
	public float[] ProductName { get; set; }

	[ColumnName(@"Description")]
	public float[] Description { get; set; }

	[ColumnName(@"Features")]
	public float[] Features { get; set; }

	[ColumnName(@"PredictedLabel")]
	public string PredictedLabel { get; set; }

	[ColumnName(@"Score")]
	public float[] Score { get; set; }

}