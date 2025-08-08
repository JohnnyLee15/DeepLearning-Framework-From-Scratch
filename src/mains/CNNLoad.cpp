    // Pipeline pipe = Pipeline::loadFromBin("modelTest");
    // ImageData2D data = *dynamic_cast<ImageData2D*>(pipe.getData());
    // ImageTransform2D transformer = *pipe.getImageTransformer();
    // NeuralNet *nn = pipe.getModel();

    // data.readTrain(trainPath);
    // data.readTest(testPath);

    // Tensor xTrain = transformer.transform(data.getTrainFeatures());
    // Tensor xTest = transformer.transform(data.getTestFeatures());
    // vector<float> yTrain = data.getTrainTargets();
    // vector<float> yTest = data.getTestTargets();

    // ProgressMetric *metric = new ProgressAccuracy(data.getNumTrainSamples());

    // nn->fit(
    //     xTrain,
    //     yTrain,
    //     0.0001,
    //     0.0001,
    //     30,
    //     8,
    //     *metric
    // );

    // pipe.saveToBin("penumoniaXrayClassifier2");