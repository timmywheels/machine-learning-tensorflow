require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')


function knn(features, labels, predictionPoint, k = 2) {
    return features
        .sub(predictionPoint) // subtract prediction from each lat/long to get distance
        .pow(2) // square values
        .sum(1) // sum values for row
        .pow(.5) // get square root of each
        .expandDims(1) // convert to 2D tensor
        .concat(labels, 1) // add labels
        .unstack() // convert to js array primitive
        .sort((a, b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1) // sort primitive js array by distance to prediction point
        .slice(0, k) // get top k records
        .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k // sum values from the label column, then divide by k for average / predicted price
}

let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long'],
    labelColumns: ['price']
})

// console.log('testFeatures:', testFeatures)
// console.log('testLabels:', testLabels)

features = tf.tensor(features)
labels = tf.tensor(labels)
testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10)
    const err = (((testLabels[i][0] - result) / testLabels[i][0]) * 100).toFixed(2)
    console.log(`Accuracy: ${err}%`)
    console.log(`Prediction: ${result} -> Actual: ${testLabels[i][0]}`)
    console.log("=========================")
})
