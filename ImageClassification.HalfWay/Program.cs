using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;
using System.Data;

var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var assetsRelativePath = Path.Combine(projectDirectory, "assets");
var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
var testRelativePath = Path.Combine(projectDirectory, "test");

Console.WriteLine("Press A to train and save a model, press B to make predicitions with a saved model");

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

	// Ladda in data från bilderna i assets-mappen
	IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

	// Ladda in bilderna i en IDataView
	IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

	// Gör en preproccessing-pipeline
	IDataView preProcessedData = GetPreProcessedData(mlContext, imageData);

	// Dela upp i test- och träningsdata
	TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);

	// Dela upp testsettet i validerings- och testdata (behövs i options nedan)
	TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

	IDataView trainSet = null; // TODO: Fyll i från trainSplit!
	IDataView validationSet = null; // TODO: Fyll i från validationTestSplit!
	IDataView testSet = null; // TODO: Fyll i från validationTestSplit

	// Här har jag kopierat från  exempelkoden här: https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification-api-transfer-learning
	// Du kan testa att variera dessa options!
	var classifierOptions = new ImageClassificationTrainer.Options()
	{
		FeatureColumnName = "Image",
		LabelColumnName = "LabelAsKey",
		ValidationSet = validationSet,
		Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
		MetricsCallback = (metrics) => Console.WriteLine(metrics),
		TestOnTrainSet = false,
		ReuseTrainSetBottleneckCachedValues = true,
		ReuseValidationSetBottleneckCachedValues = true
	};

	// Skapa pipeline utifrån classifierOptions
	var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
		.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

	// Träna modellen på träningsdata
	ITransformer trainedModel = trainingPipeline.Fit(trainSet);

	// Spara modellen i workspace-mappen
	SaveModel(mlContext, trainedModel, trainSet);

	// Utvärdera modellen
	IDataView predictions = trainedModel.Transform(testSet);
	MulticlassClassificationMetrics metrics =
		mlContext.MulticlassClassification.Evaluate(predictions,
			labelColumnName: "LabelAsKey",
			predictedLabelColumnName: "PredictedLabel");

	// Printa ut några metrics till consolen!
}

IDataView GetPreProcessedData(MLContext mlContext, IDataView data)
{
	var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
			inputColumnName: "Label",
			outputColumnName: "LabelAsKey")
		.Append(mlContext.Transforms.LoadRawImageBytes(
			outputColumnName: "Image",
			imageFolder: assetsRelativePath,
			inputColumnName: "ImagePath"));

	return preprocessingPipeline
		.Fit(data)
		.Transform(data);
}

void SaveModel(MLContext mlContext, ITransformer trainedModel, IDataView data)
{
	// Save Trained Model
	mlContext.Model.Save(trainedModel, data.Schema, Path.Combine(workspaceRelativePath, "model.zip"));
}

static void OutputPrediction(ModelOutput prediction)
{
	string imageName = Path.GetFileName(prediction.ImagePath);
	Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel} | Score: {prediction.Score.Max()}");
}

IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
{
	var files = Directory.GetFiles(folder, "*",
	searchOption: SearchOption.AllDirectories);

	foreach (var file in files)
	{
		if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
			continue;

		var label = Path.GetFileName(file);

		if (useFolderNameAsLabel)
			label = Directory.GetParent(file).Name;
		else
		{
			for (int index = 0; index < label.Length; index++)
			{
				if (!char.IsLetter(label[index]))
				{
					label = label.Substring(0, index);
					break;
				}
			}
		}

		yield return new ImageData()
		{
			ImagePath = file,
			Label = label
		};
	}
}

void MakePredictions()
{
	MLContext mlContext = new MLContext();

	// Ladda in modellen
	ITransformer trainedModel = mlContext.Model.Load(Path.Combine(workspaceRelativePath, "model.zip"), out _);

	// Ladda in bilder från test-mappen
	IEnumerable<ImageData> images = LoadImagesFromDirectory(testRelativePath, useFolderNameAsLabel: true);
	IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

	IDataView preProcessedData = GetPreProcessedData(mlContext, imageData);

	IEnumerable<ModelInput> imageInputs = mlContext.Data.CreateEnumerable<ModelInput>(preProcessedData, reuseRowObject: true);

	var predictor = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

	Console.WriteLine("Classifying multiple images");
	// Gör förutsägelser genom att loopa igenom imageInputs och anropa OutputPrediction för varje prediction
	// Ledtråd: använd metoden predict i predictor!
	
}

public class ImageData
{
	public string ImagePath { get; set; }

	public string Label { get; set; }
}

class ModelInput
{
	public byte[] Image { get; set; }

	public UInt32 LabelAsKey { get; set; }

	public string ImagePath { get; set; }

	public string Label { get; set; }
}

public class ModelOutput
{
	public string ImagePath { get; set; }

	public string Label { get; set; }

	public string PredictedLabel { get; set; }

	public float[]? Score { get; set; }
}