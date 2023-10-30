import Jimp from 'jimp';
import { Tensor } from 'onnxruntime-web';

// Function to load an image, resize it, and convert it into a tensor
export async function getImageTensorFromPath(path, dims = [1, 3, 224, 224]) {
  // 1. Load the image
  var image = await loadImageFromPath(path, dims[2], dims[3]);

  // 2. Convert the image to a tensor
  var imageTensor = imageDataToTensor(image, dims);

  // 3. Return the tensor
  return imageTensor;
}

// Function to load an image using Jimp
async function loadImageFromPath(path, width = 224, height = 224) {
  // Use Jimp to load the image and resize it.
  var imageData = await Jimp.read(path).then((imageBuffer) => {
    return imageBuffer.resize(width, height);
  });

  return imageData;
}

// Function to convert image data into a tensor
function imageDataToTensor(image, dims) {
  // 1. Get buffer data from the image and create R, G, and B arrays.
  var imageBufferData = image.bitmap.data;
  const [redArray, greenArray, blueArray] = [[], [], []];

  // 2. Loop through the image buffer and extract the R, G, and B channels
  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(imageBufferData[i]);
    greenArray.push(imageBufferData[i + 1]);
    blueArray.push(imageBufferData[i + 2]);
    // Skip data[i + 3] to filter out the alpha channel
  }

  // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
  const transposedData = redArray.concat(greenArray).concat(blueArray);

  // 4. Convert to float32
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (let i = 0, l = transposedData.length; i < l; i++) {
    float32Data[i] = transposedData[i] / 255.0; // Convert to float
  }

  // 5. Create the tensor object from onnxruntime-web
  const inputTensor = new Tensor("float32", float32Data, dims);
  return inputTensor;
}
