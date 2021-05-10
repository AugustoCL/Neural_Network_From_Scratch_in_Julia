using Statistics
using Flux.Data.MNIST
using Flux: Chain, Dense, train!, params, params!, Descent, throttle, @epochs, onecold, onehot, update!, relu, softmax
using Flux.Losses: crossentropy
using Flux.Data: DataLoader

# loading data
labels = MNIST.labels();
images = MNIST.images();

# preparing data
preprocess(img) = Float64.(img)[:]

## train data
r_train = 1:48000
x_train = hcat( preprocess.(images[r_train])... )
y_train = hcat( [ [i ≠ j ? 0.0 : 1.0 for j in 0:9] for i in labels[r_train] ]... )
train = DataLoader( (x_train, y_train), batchsize = 2400, shuffle = true)

## train data
r_test = 48001:60000
x_test = hcat( preprocess.(images[r_test])... )
y_test = hcat( [ [i ≠ j ? 0.0 : 1.0 for j in 0:9] for i in labels[r_test] ]... )
test = (x_test, y_test)


# setup model
m = Chain(
    Dense(784, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 10), 
    softmax)

loss(x, y) = Flux.Losses.mse(m(x), y)
# loss(x, y) = crossentropy(m(x), y)
ps = params(m)
opt = Descent()

# one iteration train with 20 batchs each  
train!(loss, ps, train, opt)
(loss(x_train, y_train), loss(test...))

# prepare multiple iterations train with 20 batchs each
evalcb() = @show(loss(x_train, y_train))
throtle_cb = throttle(evalcb, 3)
@time @epochs 5 train!(loss, ps, train, opt, cb = throtle_cb)

# using custom callback function
function upd_loss()
    loss_train = loss(x_train, y_train)
    loss_test = loss(x_test, y_test)
    push!(parameters, params(m))
    push!(train_loss, loss_train)
    push!(test_loss, loss_test)
    println("Train loss: $(round(loss_train,digits=6)) | Test loss: $(round(loss_test,digits=6))")
end

train_loss = Float64[];
test_loss = Float64[];
parameters = [];

throtle_cb1 = throttle(upd_loss, 1)
@epochs 20 train!(loss, ps, train, opt, cb = throtle_cb1)

# ploting losses
using Plots
plot(collect(1:length(train_loss)), [train_loss test_loss], labels = ["Train Loss" "Test Loss"])

# acuracia
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y)) # cute way to find average of correct guesses
accuracy(x_train, y_train)
accuracy(x_test, y_test)

# w1, b1, w2, b2, w3, b3 = parameters[20];
