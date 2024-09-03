module VAE

using Flux
using JLD2
using MLDatasets
using Statistics
using Random
using Images
using ImageView


# Load MNIST data
train_X, train_y = MNIST.traindata(Float32)
train_X = train_X ./ 255.0  # Normalize to [0, 1]

# Reshape images to 4D tensor: (Height, Width, Channels, Batch)
train_X = reshape(train_X, 28, 28, 1, :)

# Define the autoencoder with Conv layers
latent_h = 7
latent_w = 7
latent_channels = 8  # Number of channels in the latent space

encoder = Chain(
    Conv((4, 4), 1 => 16, stride=2, pad=1),
    BatchNorm(16),
    relu,
    Conv((4, 4), 16 => 32, stride=2, pad=1),
    BatchNorm(32),                  
    relu,      
    Conv((3, 3), 32 => 16, stride=1, pad=1)
)

decoder = Chain(
    ConvTranspose((3, 3), 16 => 32, stride=1, pad=1),
    BatchNorm(32),
    relu,
    ConvTranspose((4, 4), 32 => 16, stride=2, pad=1),
    BatchNorm(16),
    relu,
    ConvTranspose((4, 4), 16 => 1, stride=2, pad=1),
    sigmoid
)



autoencoder = Chain(encoder, decoder)
#@load "autoencoder.jld2" autoencoder

# Mean Squared Error loss function
loss(x) = Flux.Losses.mse(autoencoder(x), x)

# Hyperparameters
learning_rate = 1e-3
num_epochs = 50
batch_size = 512

# Optimizer
opt = Flux.ADAM(learning_rate)

# Training loop
for epoch in 1:num_epochs
    for i in 1:batch_size:size(train_X, 4)
        
        #println(i)
        batch = train_X[:, :, :, i:min(i + batch_size - 1, size(train_X, 4))]
        
        # Compute the gradients
        gs = Flux.gradient(() -> loss(batch), Flux.params(autoencoder))
        
        # Update the model parameters
        Flux.Optimise.update!(opt, Flux.params(autoencoder), gs)
    end
    
    println("Epoch $epoch complete, Loss: $(loss(train_X))")
end

# Save model
@save "autoencoder.jld2" autoencoder=autoencoder



# Load test data
test_X, _ = MNIST.testdata(Float32)
test_X = test_X ./ 255.0
test_X = reshape(test_X, 28, 28, 1, :)

# Reconstruct some test images
reconstructed_X = autoencoder(test_X[:, :, :, 1:10])

# Convert to images and display
for i in 1:3
    original_img = reshape(test_X[:, :, 1, i], 28, 28)
    reconstructed_img = reshape(reconstructed_X[:, :, 1, i], 28, 28)
    
    println("Original Image $i:")
    imshow(Gray.(original_img))
    
    println("Reconstructed Image $i:")
    imshow(Gray.(reconstructed_img))
end

end # module