using Flux
using MLDatasets: MNIST
using BSON: @load

tensor_train, labels_train = MNIST.traindata(Float64)
tensor_test, labels_test = MNIST.testdata(Float64)

xtrain = reshape(tensor_train, 784, :)
xtest = reshape(tensor_test, 784, :)

@load "MNIST_v2.bson" model

function prediction(x)
    pred_vector = model(x) 
    pred_val = findfirst(x -> x == maximum(pred_vector), pred_vector) - 1
    return pred_val
end

prediction(xtest[:, 10])

labels_test[10]