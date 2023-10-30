import * as ort from 'onnxruntime-web';
import _ from 'lodash';
import { imagenetClasses } from '../data/imagenet';

// Function to run inference using a Squeezenet model
export async function runSqueezenetModel(preprocessedData) {
  // Create an inference session and set options
  const session = await ort.InferenceSession.create(
    './_next/static/chunks/pages/squeezenet1_1.onnx',
    { executionProviders: ['webgl'], graphOptimizationLevel: 'all' }
  );
  console.log('Inference session created');

  // Run inference and get results
  const [results, inferenceTime] = await runInference(session, preprocessedData);
  return [results, inferenceTime];
}

// Function to run inference using a given session
async function runInference(session, preprocessedData) {
  // Get start time to calculate inference time
  const start = new Date();

  // Create feeds with the input name from the model export and the preprocessed data
  const feeds = {};
  feeds[session.inputNames[0]] = preprocessedData;

  // Run the session inference
  const outputData = await session.run(feeds);

  // Get the end time to calculate inference time
  const end = new Date();

  // Convert to seconds
  const inferenceTime = (end.getTime() - start.getTime()) / 1000;

  // Get output results with the output name from the model export
  const output = outputData[session.outputNames[0]];

  // Get the softmax of the output data. The softmax transforms values to be between 0 and 1
  var outputSoftmax = softmax(Array.prototype.slice.call(output.data));

  // Get the top 5 results
  var results = imagenetClassesTopK(outputSoftmax, 5);
  console.log('results: ', results);

  return [results, inferenceTime];
}

// The softmax function transforms values to be between 0 and 1
function softmax(resultArray) {
  // Get the largest value in the array
  const largestNumber = Math.max(...resultArray);

  // Apply the exponential function to each result item subtracted by the largest number
  // Use reduce to get the previous result number and the current number to sum all the exponential results
  const sumOfExp = resultArray.map((resultItem) =>
    Math.exp(resultItem - largestNumber)
  ).reduce((prevNumber, currentNumber) => prevNumber + currentNumber);

  // Normalize the resultArray by dividing by the sum of all exponentials
  // This normalization ensures that the sum of the components of the output vector is 1
  return resultArray.map((resultValue, index) =>
    Math.exp(resultValue - largestNumber) / sumOfExp
  );
}

// Function to find the top-k imagenet classes
export function imagenetClassesTopK(classProbabilities, k = 5) {
  const probs = _.isTypedArray(classProbabilities)
    ? Array.prototype.slice.call(classProbabilities)
    : classProbabilities;

  const sorted = _.reverse(
    _.sortBy(
      probs.map((prob, index) => [prob, index]),
      (probIndex) => probIndex[0]
    )
  );

  const topK = _.take(sorted, k).map((probIndex) => {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1].toString(), 10),
      name: iClass[1].replace(/_/g, ' '),
      probability: probIndex[0],
    };
  });

  return topK;
}
