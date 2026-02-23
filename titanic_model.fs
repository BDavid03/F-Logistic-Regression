namespace TitanicPrediction

open System
open System.IO
open System.Globalization
open FSharp.Data

/// Small "Closed Worlds" make bugs cry
type Sex =
    | Male
    | Female

type Embarked =
    | C
    | Q
    | S

type Pclass =
    | First
    | Second
    | Third

type Passenger =
    { PassengerId : int
      Survived    : bool option
      Pclass      : Pclass
      Name        : string
      Sex         : Sex
      Age         : float option
      SibSp       : int
      Parch       : int
      Ticket      : string
      Fare        : float option
      Cabin       : string option
      Embarked    : Embarked option }

type TrainCsv =
    CsvProvider<"Samples/train.csv", HasHeaders=true, ResolutionFolder=__SOURCE_DIRECTORY__>

type TestCsv =
    CsvProvider<"Samples/test.csv", HasHeaders=true, ResolutionFolder=__SOURCE_DIRECTORY__>

module Parse =

    let toSex = function
        | "male" -> Male
        | "female" -> Female
        | x -> failwith $"Unknown Sex: {x}"

    let toEmbarked = function
        | "C" -> Some C
        | "Q" -> Some Q
        | "S" -> Some S
        | ""  -> None
        | x   -> failwith $"Unknown Embarked: {x}"

    let toPclass = function
        | 1 -> First
        | 2 -> Second
        | 3 -> Third
        | x -> failwith $"Unknown Pclass: {x}"

    let optFloat (v: Nullable<float>) =
        if v.HasValue then Some v.Value else None

    let optString (s: string) =
        if String.IsNullOrWhiteSpace s then None else Some s

    let optFloatFromUnknown (x: obj) : float option =
        match x with
        | null -> None
        | :? Nullable<float> as v when v.HasValue -> Some v.Value
        | :? float as f -> Some f
        | :? Nullable<decimal> as v when v.HasValue -> Some (float v.Value)
        | :? decimal as d -> Some (float d)
        | :? string as s when not (String.IsNullOrWhiteSpace s) ->
            // Titanic CSVs are dot-decimal; force invariant culture
            Some (float (Decimal.Parse(s, CultureInfo.InvariantCulture)))
        | _ -> None

    let toSurvived (x: obj) : bool =
        match x with
        | :? int as i -> i = 1
        | :? bool as b -> b
        | :? string as s ->
            match s.Trim() with
            | "1" -> true
            | "0" -> false
            | "true" | "True" -> true
            | "false" | "False" -> false
            | _ -> failwith $"Unexpected Survived string: {s}"
        | _ -> failwith "Unexpected Survived type"

    let toPassengerFromTrain (r: TrainCsv.Row) : Passenger =
        { PassengerId = r.PassengerId
          Survived    = Some (toSurvived (box r.Survived))
          Pclass      = toPclass r.Pclass
          Name        = r.Name
          Sex         = toSex r.Sex
          Age         = optFloat r.Age
          SibSp       = r.SibSp
          Parch       = r.Parch
          Ticket      = r.Ticket
          Fare        = optFloatFromUnknown (box r.Fare)
          Cabin       = optString r.Cabin
          Embarked    = toEmbarked r.Embarked }

    let toPassengerFromTest (r: TestCsv.Row) : Passenger =
        { PassengerId = r.PassengerId
          Survived    = None
          Pclass      = toPclass r.Pclass
          Name        = r.Name
          Sex         = toSex r.Sex
          Age         = optFloat r.Age
          SibSp       = r.SibSp
          Parch       = r.Parch
          Ticket      = r.Ticket
          Fare        = optFloatFromUnknown (box r.Fare)
          Cabin       = optString r.Cabin
          Embarked    = toEmbarked r.Embarked }

module Data =

    let samplesDir = Path.Combine(__SOURCE_DIRECTORY__, "Samples")
    let trainPath  = Path.Combine(samplesDir, "train.csv")
    let testPath   = Path.Combine(samplesDir, "test.csv")

    let loadTrain () =
        TrainCsv.Load(trainPath).Rows
        |> Seq.map Parse.toPassengerFromTrain
        |> Seq.toList

    let loadTest () =
        TestCsv.Load(testPath).Rows
        |> Seq.map Parse.toPassengerFromTest
        |> Seq.toList

module Features =

    /// ML.NET likes mutable classes with get/set.
    /// We give it a single vector column named "Features" to avoid Concatenate typing issues in F#.
    open Microsoft.ML.Data
    [<CLIMutable>]
    type ModelInput =
        { Label : bool
          [<VectorType(9)>]
          Features : float32[] }

    let private sexTo01 = function
        | Male -> 1.0
        | Female -> 0.0

    let private pclassToFloat = function
        | First -> 1.0
        | Second -> 2.0
        | Third -> 3.0

    let private embarkedOneHot = function
        | Some C -> 1.0, 0.0, 0.0
        | Some Q -> 0.0, 1.0, 0.0
        | Some S -> 0.0, 0.0, 1.0
        | None   -> 0.0, 0.0, 0.0

    /// Turn a Passenger into model-ready input.
    /// Missing values get simple defaults (0.0).
    let toInput (p: Passenger) : ModelInput =
        let eC, eQ, eS = embarkedOneHot p.Embarked

        let feats : float32[] =
            [| float32 (pclassToFloat p.Pclass)
               float32 (sexTo01 p.Sex)
               float32 (defaultArg p.Age 0.0)
               float32 p.SibSp
               float32 p.Parch
               float32 (defaultArg p.Fare 0.0)
               float32 eC
               float32 eQ
               float32 eS |]

        { Label = defaultArg p.Survived false
          Features = feats }

module Model =

    open Microsoft.ML
    open Microsoft.ML.Data

    [<CLIMutable>]
    type ModelOutput =
        { [<ColumnName("PredictedLabel")>]
          Predicted   : bool
          Probability : float32
          Score       : float32 }

    let train (trainPassengers: Passenger list) =
        let ml = MLContext(seed = Nullable 1)

        let inputs =
            trainPassengers
            |> List.choose (fun p -> p.Survived |> Option.map (fun _ -> Features.toInput p))

        let data = ml.Data.LoadFromEnumerable(inputs)

        let trainer =
            ml.BinaryClassification.Trainers.LbfgsLogisticRegression(
                labelColumnName = "Label",
                featureColumnName = "Features")

        let model = trainer.Fit(data)
        model, ml

    let evaluateWithSplit (ml: MLContext) (trainPassengers: Passenger list) =
        let inputs =
            trainPassengers
            |> List.choose (fun p -> p.Survived |> Option.map (fun _ -> Features.toInput p))

        let data = ml.Data.LoadFromEnumerable(inputs)
        let split = ml.Data.TrainTestSplit(data, testFraction = 0.2)

        let trainer =
            ml.BinaryClassification.Trainers.LbfgsLogisticRegression(
                labelColumnName = "Label",
                featureColumnName = "Features")

        let model = trainer.Fit(split.TrainSet)
        let preds = model.Transform(split.TestSet)
        ml.BinaryClassification.Evaluate(preds, labelColumnName = "Label")

    let predict (model: ITransformer, ml: MLContext) (testPassengers: Passenger list) =
        let engine = ml.Model.CreatePredictionEngine<Features.ModelInput, ModelOutput>(model)

        testPassengers
        |> List.map (fun p ->
            let output = engine.Predict(Features.toInput p)
            p.PassengerId, output.Predicted, output.Probability)

module Runner =

    open System.Diagnostics

    let run () =
        let sw = Stopwatch.StartNew()

        // 1) Load data
        let trainPassengers = Data.loadTrain()
        let testPassengers  = Data.loadTest()

        // 2) Train model
        let model, ml = Model.train trainPassengers

        // 3) Evaluate (train/test split)
        let metrics = Model.evaluateWithSplit ml trainPassengers

        // 4) Predict on test set (first few)
        let preds = Model.predict (model, ml) testPassengers

        sw.Stop()

        printfn "Train rows: %d" trainPassengers.Length
        printfn "Test rows : %d" testPassengers.Length
        printfn "Accuracy  : %f" metrics.Accuracy
        printfn "AUC       : %f" metrics.AreaUnderRocCurve
        printfn "Time (ms) : %d" sw.ElapsedMilliseconds
        printfn "First 5 predictions: %A" (preds |> List.truncate 40)

        0
