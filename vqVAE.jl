module VAE

using Flux
using Zygote
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
latent_channels = 16  # Number of channels in the latent space

encoder = Chain(
    Conv((4, 4), 1 => 16, stride=2, pad=1),
    BatchNorm(16),
    relu,
    Conv((4, 4), 16 => 32, stride=2, pad=1),
    BatchNorm(32),                  
    relu,      
    Conv((3, 3), 32 => latent_channels, stride=1, pad=1),
    BatchNorm(latent_channels),   
)

decoder = Chain(
    ConvTranspose((3, 3), latent_channels => 32, stride=1, pad=1),
    BatchNorm(32),
    relu,
    ConvTranspose((4, 4), 32 => 16, stride=2, pad=1),
    BatchNorm(16),
    relu,
    ConvTranspose((4, 4), 16 => 1, stride=2, pad=1),
    sigmoid
)

function stop_gradient(x)
    Zygote.ignore() do
        return x
    end
end

mutable struct VectorQuantizer
    codebook::Matrix{Float32}  # Codebook: vector of embedding vectors

    function VectorQuantizer(latent_channels, codebook_length)
        # Initialize the codebook with random vectors
        codebook = rand(Float32, codebook_length, latent_channels) * 2 .- 1
        new(codebook)
    end
    
    function quantizeVec(vq, v) 
        distances = [sum(abs.(v - cb_vec)) for (i, cb_vec) in enumerate(vq.codebook)]
        return vq.codebook[argmin(distances)]
    end

    # Forward pass: z is (latent_h, latent_w, latent_channels, batch_size)
    function (vq::VectorQuantizer)(z::Array{Float32, 4})
        H, W, C, B = size(z)
    
        # Flatten the input: (C, H*W*B)
        z_flatten = reshape(permutedims(z, (3, 1, 2, 4)), C, H*W*B)
    
        codebook = vq.codebook
    
        # Squared norms of the input vectors: (H*W*B, 1)
        input_squared = sum(z_flatten.^2, dims=1)
    
        # Squared norms of the codebook vectors: (codebook_length, 1)
        codebook_squared = sum(codebook.^2, dims=2)
    
        # Inner product between input vectors and codebook vectors
        # z_flatten: (C, H*W*B) and codebook: (codebook_length, C)
        # result: (codebook_length, H*W*B)
        inner_product = codebook * z_flatten
    
        # Compute distances:
        distances = input_squared .+ codebook_squared .- 2 * inner_product
    
        # Find the index of the closest codebook vector for each input vector
        min_indices = argmin(distances, dims=1)[:]
        min_indices_vector = [index[1] for index in min_indices]
    
        #select codebook vectors by min_indices
        z_q = codebook[min_indices_vector, :]
      
    
        # Reshape the quantized vectors back to the original shape (C, H, W, B)
        z_q_reshaped = reshape(z_q', C, H, W, B)
    
        # Return z + stop_gradient(z_q_reshaped - z)
        return permutedims(z_q_reshaped, (2,3,1,4))
    end
end

Flux.@functor VectorQuantizer

# Loss components
function reconstruction_loss(x, x̂)
    return Flux.Losses.mse(x̂, x)
end

function vq_loss(z, z_q, β=0.025)
    # Codebook loss: moves the codebook vectors toward the encoder output (z)
    codebook_loss = Flux.Losses.mse(stop_gradient(z), z_q)
    
    # Commitment loss: encourages the encoder output (z) to commit to the quantized values (z_q)
    commitment_loss = β * Flux.Losses.mse(z, stop_gradient(z_q))
    
    return codebook_loss + commitment_loss
end

# Full loss function: combining reconstruction loss and vector quantization loss
function loss(x)
    # Forward pass through encoder, vector quantizer, and decoder
    z = encoder(x)
    z_q = vq(z)  # Quantize the latent space with the vector quantizer
    x̂ = decoder(z + stop_gradient(z_q - z))  # Decode the quantized latent space
    
    # Compute the different losses
    rec_loss = reconstruction_loss(x, x̂)  # Reconstruction loss
    
    quantizer_loss = vq_loss(z, z_q)     # Vector quantization + commitment loss
    
    return rec_loss + quantizer_loss * 0.1, rec_loss, quantizer_loss
end

# Initialize Vector Quantizer and Optimizer
latent_channels = 16
vq = VectorQuantizer(latent_channels, 100)

@load "autoencoder.jld2" autoencoder
encoder, decoder, vq = autoencoder

# Optimizer
learning_rate = 1e-4
opt = Flux.ADAM(learning_rate)

# Hyperparameters
num_epochs = 10
batch_size = 512

println(loss(train_X[:,:,:,1:1000]))

# Training loop
for epoch in 1:num_epochs
    for i in 1:batch_size:size(train_X, 4)
         
        #println(i)
        batch = train_X[:, :, :, i:min(i + batch_size - 1, size(train_X, 4))]
        params = Flux.params(vq)
        
        
        # Compute the gradients with respect to the combined loss
        
        gs = Flux.gradient(() -> (loss(batch)[1]), params)
        
        # Update the model parameters (including the codebook)
        Flux.Optimise.update!(opt, params, gs)
        
    end

    total_loss, rec_loss, vq_loss = loss(train_X)
    println("Epoch $epoch complete, Loss: $(total_loss)")
    println("Reconstruction Loss: $(rec_loss)")
    println("VQ Loss: $(vq_loss)")
    
    
end

# Save model
@save "autoencoder.jld2" autoencoder=(encoder, decoder, vq)

end # module