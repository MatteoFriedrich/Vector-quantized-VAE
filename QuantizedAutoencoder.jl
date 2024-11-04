module VAE

using Flux
using Zygote
using JLD2
using MLDatasets
using Statistics
using Random
using Images
using ImageView
using PyPlot
using LinearAlgebra

# Load MNIST data
train_X, train_y = MNIST.traindata(Float32)
train_X = train_X ./ 255.0  # Normalize to [0, 1]

# Reshape images to 4D tensor: (Height, Width, Channels, Batch)
train_X = reshape(train_X, 28, 28, 1, :)

encoder = Chain(
    Conv((4, 4), 1 => 24, stride=2, pad=1),
    BatchNorm(24),
    relu,
    Conv((3, 3), 24 => 32, stride=2, pad=1),
    BatchNorm(32),                  
    relu,      
    Conv((3, 3), 32 => 24, stride=2, pad=1),
    BatchNorm(24), 
    relu, 
    Conv((3, 3), 24 => 12, stride=1, pad=1),
    BatchNorm(12), 
    relu, 
    Flux.flatten,
    Dense(192, 50, tanh),
    Dense(50,30, tanh),
    Dense(30,20, tanh),
    Dense(20,12, tanh),
    sigmoid
)

decoder = Chain(
    Dense(12, 20, tanh),
    Dense(20, 30, tanh),
    Dense(30, 50, tanh),
    Dense(50, 192, tanh),
    x -> reshape(x, (4,4,12,:)),
    ConvTranspose((3, 3), 12 => 16, stride=1, pad=1),
    BatchNorm(16),
    relu,
    ConvTranspose((3, 3), 16 => 24, stride=2, pad=1),
    BatchNorm(24),
    relu,
    ConvTranspose((4, 4), 24 => 16, stride=2, pad=1),
    BatchNorm(16),
    relu,
    ConvTranspose((4, 4), 16 => 1, stride=2, pad=1),
    sigmoid
)

function stop_gradient(x)
    return Zygote.ignore() do
        return x
    end
end

# Loss components
function reconstruction_loss(x, x̂)
    return Flux.Losses.mse(x̂, x)
end

function pairwise_log_distance(z)
    # z is a matrix where each column is a vector.
    n = size(z, 2)  # Number of column vectors

    # Compute pairwise squared distances between columns
    col_norms = sum(z.^2, dims=1)       # Squared norms of each column (z_i . z_i)
    cross_term = z' * z                 # Dot product between columns (z_i . z_j)

    # Pairwise squared distances ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 * (z_i . z_j)
    pairwise_sq_dist = col_norms .+ col_norms' .- 2 * cross_term
    

    # Take the square root to get Euclidean distances and add a small constant to avoid log(0)
    epsilon = 1e-4
    pairwise_dist = sqrt.(pairwise_sq_dist .+ epsilon)
    norm_pairwise_dist = pairwise_dist ./ sqrt(size(z,1))
    


    # Compute the log of each pairwise distance
    log_distances = log.(norm_pairwise_dist .+ 1e-8)
    #println(log_distances)

    # Mean log distance: sum of log distances / number of pairs
    avg_log_distance = sum(log_distances) / n^2  # Average over all column pairs

    return avg_log_distance
end

function evaluate()
    n = abs(rand(Int)) % size(train_X,4)
    subplot(1,2,1)
    PyPlot.imshow(train_X[:,:,1,n]')

    subplot(1,2,2)
    reconstructed = decoder(round.(encoder(train_X[:,:,:,n:n])))
    PyPlot.imshow(reconstructed[:,:,1,1]')
end

function encoder_loss(x, show_quantized=false)
    z = encoder(x)
    #z_q = z + stop_gradient(round.(z) - z)
    z_q = round.(z)
    x̂ = decoder(z)
    
    rec_loss = reconstruction_loss(x, x̂)

    quant_loss = reconstruction_loss(z, z_q)
    distance_loss = pairwise_log_distance(z[:,1:32])

    return rec_loss - 1e-7 * distance_loss, rec_loss, quant_loss, distance_loss

    return reconstruction_loss(x, decoder(encoder(x)))
end

function decoder_loss(x, show_quantized=false)
    #=z = encoder(x)
    rec_loss = reconstruction_loss(x, decoder(z))
    quant_rec_loss = reconstruction_loss(x, decoder(round.(z)))
    
    return rec_loss + quant_rec_loss, rec_loss, quant_rec_loss=#
    return reconstruction_loss(x, decoder(round.(encoder(x))))
end

@load "autoencoder.jld2" autoencoder
#@load "models/quant_autoencoder_12d_61E-8.jld2" autoencoder

encoder, decoder = autoencoder

# Optimizer
learning_rate = 1e-4
opt = Flux.ADAM(learning_rate)

# Hyperparameters
num_epochs = 20
batch_size = 32

println(encoder_loss(train_X[:,:,:,1:1000]))
#println(decoder_loss(train_X[:,:,:,:]))

# Training loop
for epoch in 1:num_epochs
    for i in 1:batch_size:size(train_X, 4)
         
        #println(i)
        batch = train_X[:, :, :, i:min(i + batch_size - 1, size(train_X, 4))]
        dec_params = Flux.params(decoder) 
        #enc_params = Flux.params(encoder) 
        
         
        # Compute the gradients with respect to the combined loss
        
        decoder_gs = Flux.gradient(() -> (decoder_loss(batch)[1]), dec_params)
        #encoder_gs = Flux.gradient(() -> (encoder_loss(batch)[1]), enc_params)
        
        # Update the model parameters (including the codebook)
        Flux.Optimise.update!(opt, dec_params, decoder_gs)
        #Flux.Optimise.update!(opt, enc_params, encoder_gs)
        
    end

    #total_loss = encoder_loss(train_X, true)
    #println("Epoch $epoch complete, Loss: $(total_loss)")  
    total_loss = decoder_loss(train_X)
    println("Epoch $epoch complete, Loss: $(total_loss)")  
    evaluate()
end

# Save model
@save "autoencoder.jld2" autoencoder=(encoder, decoder)

end # module