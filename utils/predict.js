// Language: JavaScript
// Path: react-next/utils/predict.js
const { getImageTensorFromPath } = require('./imageHelper');
const { runSqueezenetModel } = require('./modelHelper');

async function inferenceSqueezenet(path) {
  // 1. Convert image to tensor
  const imageTensor = await getImageTensorFromPath(path);
  // 2. Run model
  const [predictions, inferenceTime] = await runSqueezenetModel(imageTensor);
  // 3. Return predictions and the amount of time it took to inference.
  return [predictions, inferenceTime];
}

module.exports = {
  inferenceSqueezenet
};
