import { useRef, useState } from 'react';
import { IMAGE_URLS } from '../data/sample-image-urls';
import { inferenceSqueezenet } from '../utils/predict';
import styles from '../styles/Home.module.css';

const ImageCanvas = (props) => {
  const canvasRef = useRef(null);
  var image;
  const [topResultLabel, setLabel] = useState("");
  const [topResultConfidence, setConfidence] = useState("");
  const [inferenceTime, setInferenceTime] = useState("");

  const getImage = () => {
    var sampleImageUrls = IMAGE_URLS;
    var random = Math.floor(Math.random() * (9 - 0 + 1) + 0);
    return sampleImageUrls[random];
  };

  const displayImageAndRunInference = () => {
    image = new Image();
    var sampleImage = getImage();
    image.src = sampleImage.value;

    setLabel(`Inferencing...`);
    setConfidence("");
    setInferenceTime("");

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    image.onload = () => {
      ctx.drawImage(image, 0, 0, props.width, props.height);
    };

    submitInference();
  };

  const submitInference = async () => {
    var [inferenceResult, inferenceTime] = await inferenceSqueezenet(image.src);
    var topResult = inferenceResult[0];

    setLabel(topResult.name.toUpperCase());
    setConfidence(topResult.probability);
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);
  };

  return (
    <>
      <button
        className={styles.grid}
        onClick={displayImageAndRunInference} >
        Run Squeezenet inference
      </button>
      <br />
      <canvas ref={canvasRef} width={props.width} height={props.height} />
      <span>{topResultLabel} {topResultConfidence}</span>
      <span>{inferenceTime}</span>
    </>
  );
};

export default ImageCanvas;
