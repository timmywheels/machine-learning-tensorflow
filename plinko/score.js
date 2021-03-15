const outputs = [];

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
    outputs.push([dropPosition, bounciness, size, bucketLabel]);
    // console.log(outputs);
}

function runAnalysis() {
    // get test and training datasets using the first testSetSize items of outputs arr
    const testSetSize = 10;
    const k = 10;

    // test a range of values for k
    // _.range(1, 20).forEach(k => {

    // iterate through individual features
    _.range(0, 3).forEach(feature => {
        const data = _.map(outputs, row => [row[feature], _.last(row)])
        const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize);
        const accuracy = _.chain(testSet)
            // iterate through testSet
            // and get all matches where predicted bucket == actual bucket
            .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint))
            // take number of accurate predictions
            .size()
            // divided by the size of the test set
            .divide(testSetSize)
            // which provides the "accuracy"
            .value()

        // console.log(`Accuracy for k = ${k}: ${accuracy}`)
        console.log(`Accuracy for feature ${feature}: ${accuracy}`)
    });

}

function knn(data, point, k) {
    const bucket = _.chain(data)
        .map(row => {
            // lodash _.initial returns all items in an
            // arr except for the last item
            // in this case it will give us all of our 'features'
            const features = _.initial(row);
            // the last item in the arr is the 'label'
            const label = _.last(row)
            return [
                distance(features, point),
                label
            ];
        })
        .sortBy(row => row[0])
        .slice(0, k)
        .countBy(row => row[1])
        .toPairs()
        .sortBy(row => row[1])
        .last()
        .first()
        .parseInt()
        .value();
    return bucket;
}

function distance(pointA, pointB) {
    // return the absolute value from any set
    // of points a & b
    // example inputs: pointA = [1, 1], pointB = [4, 5]
    return _.chain(pointA)
        // take the corresponding index from a,
        // and match it with b in a new arr pair
        .zip(pointB)
        // take the items from the new arr pair
        // subtract them, then square them
        .map(([a, b]) => (a - b) ** 2)
        // add all items in arr
        .sum()
        // return square root of sum
        .value() ** 0.5;
}

function splitDataset(data, testCount) {
    // randomize data to get a better sample set
    const shuffled = _.shuffle(data);

    // generate a test dataset with a size of testCount
    const testSet = _.slice(shuffled, 0, testCount)

    // generate a training dataset with a size of testCount
    const trainingSet = _.slice(shuffled, testCount);

    // return array containing test and training datasets
    return [testSet, trainingSet];
}

// minMax will normalize the dataset values from 0 to 1
function minMax(data, featureCount) {
    // featureCount is used to determine
    // how many columns need to be normalized
    // so that the 'label' (last index) can be ignored

    const clonedData = _.cloneDeep(data)

    // iterate over each column of features
    for (let i = 0; i < featureCount; i++) {
        const column = clonedData.map(row => row[i])

        const min = _.min(column)
        const max = _.max(column)

        for (let j = 0; j < clonedData.length; j++) {
            clonedData[j][i] = (clonedData[j][i] - min) / (max - min)
        }

    }
    return clonedData
}