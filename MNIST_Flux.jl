using Statistics, Flux, Plots
using Flux.Data.MNIST
using Flux: Chain, Dense, train!, params, params!, Descent, throttle, @epochs, onecold, onehot, update!, relu, softmax
using Flux.Losses: crossentropy
using Flux.Data: DataLoader
using BSON: @save

# loading data
labels_train = MNIST.labels();
images_train = MNIST.images();
labels_test = MNIST.labels(:test);
images_test = MNIST.images(:test);

# preparing data
preprocess(img) = Float64.(img)[:]

## train data
x_train = hcat( preprocess.(images_train)... )
y_train = hcat( [ [i ≠ j ? 0.0 : 1.0 for j in 0:9] for i in labels_train ]... )
train = DataLoader( (x_train, y_train), batchsize = 2000, shuffle = true)

## test data
x_test = hcat( preprocess.(images_test)... )
y_test = hcat( [ [i ≠ j ? 0.0 : 1.0 for j in 0:9] for i in labels_test ]... )
test = (x_test, y_test)


# setup model
m = Chain(
    Dense(784, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 10), 
    softmax)

# loss(x, y) = Flux.Losses.mse(m(x), y)
loss(x, y) = crossentropy(m(x), y)
ps = params(m)
opt = ADAM()    # Descent()

# prepare multiple iterations train with 30 batchs each
# using custom callback function
function upd_loss()
    loss_train = loss(x_train, y_train)
    acc_train = accuracy(x_train, y_train)
    loss_test = loss(x_test, y_test)
    acc_test = accuracy(x_test, y_test)
    push!(train_loss, loss_train)
    push!(test_loss, loss_test)
    push!(train_acc, acc_train)
    push!(test_acc, acc_test)
    println("Train loss: $(round(loss_train,digits=6))  | Accur. train: $(round(acc_train,digits=6))\nTest loss:  $(round(loss_test,digits=6)) | Accur. test:  $(round(acc_test,digits=6))")
    # println("Train loss: $(round(loss_train,digits=6))      | Accur. train: $(round(acc_train,digits=6))\nTest loss:  $(round(loss_test,digits=6))      | Accur. test:  $(round(acc_test,digits=6))")
end

train_loss = Float64[];
test_loss = Float64[];
train_acc = Float64[];
test_acc = Float64[];
throtle_cb1 = throttle(upd_loss, 1)
@epochs 100 train!(loss, ps, train, opt, cb = throtle_cb1)

# ploting losses
plot(collect(25:length(train_loss)), 
     [train_loss[25:end] test_loss[25:end]], 
     labels = ["Train Loss" "Test Loss"],
     lw=2,
     title = "(784, 16, 16, 10)(relu, relu, softmax)(ADAM)(100 iter)")

# ploting accuracy     
plot(collect(25:length(train_loss)), 
     [train_acc[25:end] test_acc[25:end]], 
     labels = ["Train Acc." "Test Acc."],
     lw=2,
     title = "(784, 16, 16, 10)(relu, relu, softmax)(ADAM)(100 iter)")

# acuracia
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y)) # cute way to find average of correct guesses
accuracy(x_train, y_train)
accuracy(x_test, y_test)

@save "m_100_Epochs.bson" m
# w1, b1, w2, b2, w3, b3 = parameters[20];