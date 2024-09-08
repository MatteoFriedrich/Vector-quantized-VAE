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
    Conv((3, 3), 32 => latent_channels, stride=1, pad=1)
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
    codebook::Vector{Vector{Float32}}  # Codebook: vector of embedding vectors

    function VectorQuantizer(latent_channels, codebook_length)
        # Initialize the codebook with random vectors
        codebook = [rand(Float32, latent_channels) * 2 .- 1 for _ in 1:codebook_length]
        new(codebook)
    end
    
    function quantizeVec(vq, v) 
        distances = [sum(abs.(v - cb_vec)) for (i, cb_vec) in enumerate(vq.codebook)]
        return vq.codebook[argmin(distances)]
    end

    # Forward pass: z is (latent_h, latent_w, latent_channels, batch_size)
    function (vq::VAE.VectorQuantizer)(z::Array{Float32, 4})
        H, W, C, B = size(z)
        z_q = collect([quantizeVec(vq, z[row, col, :, n]) for row = 1:H, col = 1:W, n=1:B])
        z_q_reshaped = collect(z_q[row, col, n][k] for row in 1:H, col in 1:W, k in 1:C, n in 1:B)

        return z + stop_gradient(z_q_reshaped - z), z_q_reshaped
    end
end

# Loss components
function reconstruction_loss(x, x̂)
    return Flux.Losses.mse(x̂, x)
end

function vq_loss(z, z_q, β=0.25)
    # Codebook loss: moves the codebook vectors toward the encoder output (z)
    codebook_loss = Flux.Losses.mse(stop_gradient(z), z_q)
    
    # Commitment loss: encourages the encoder output (z) to commit to the quantized values (z_q)
    commitment_loss = β * Flux.Losses.mse(z, stop_gradient(z_q))
    
    return codebook_loss + commitment_loss
end

# Full loss function: combining reconstruction loss and vector quantization loss
function loss(x)
    # Forward pass through encoder, vector quantizer, and decoder
    @time z = encoder(x)
    @time z_q, z_q_reshaped = vq(z)  # Quantize the latent space with the vector quantizer
    x̂ = decoder(z_q_reshaped)  # Decode the quantized latent space
    
    # Compute the different losses
    rec_loss = reconstruction_loss(x, x̂)  # Reconstruction loss
    vq_codebook_loss = vq_loss(z, z_q)     # Vector quantization + commitment loss
    
    return rec_loss + vq_codebook_loss
end

# Initialize Vector Quantizer and Optimizer
latent_channels = 16
vq = VectorQuantizer(latent_channels, 100)

# Optimizer
learning_rate = 1e-4
opt = Flux.ADAM(learning_rate)

# Hyperparameters
num_epochs = 1
batch_size = 16

# Training loop
for epoch in 1:num_epochs
    for i in 1:batch_size# :size(train_X, 4)
        
        println(i)
        batch = train_X[:, :, :, i:min(i + batch_size - 1, size(train_X, 4))]
        
        # Compute the gradients with respect to the combined loss
        gs = Flux.gradient(() -> loss(batch), Flux.params(encoder, decoder, vq))
        
        # Update the model parameters (including the codebook)
        Flux.Optimise.update!(opt, Flux.params(encoder, decoder, vq), gs)
    end
    
    println("Epoch $epoch complete, Loss: $(loss(train_X))")
end

# Save model
@save "autoencoder.jld2" autoencoder=(encoder, decoder, vq)

end # module