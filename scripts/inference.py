import torch
import torchvision.transforms as transforms
from PIL import Image
import spacy

from model_builder import EncodertoDecoder  # Assuming you have these defined
from data_setup import Vocabulary  # Assuming you have these defined

# Setup hyperparameters
embed_size = 256
hidden_size = 256
num_layers = 1
learning_rate = 3e-4
num_epochs = 2

load_model = False
save_model = False
train_CNN = False

# Function to preprocess input image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to generate caption
def generate_caption(image_path, encoder_decoder_model, vocabulary, device, transform):
    # Set model to evaluation mode
    encoder_decoder_model.eval()

    # Load and preprocess the image
    image = preprocess_image(image_path, transform)
    image = image.to(device)

    # Generate caption
    with torch.no_grad():
        features = encoder_decoder_model.encoderCNN(image)
        sampled_ids = encoder_decoder_model.decoderRNN.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()  # Convert to numpy array

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocabulary.idx2word[word_id]
            sampled_caption.append(word)
            if word == "<EOS>":
                break
        sentence = ' '.join(sampled_caption)

    return sentence

def main():
    # Define paths and setup
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    model_checkpoint = "path_to_your_model_checkpoint.pth.tar"  # Replace with your trained model checkpoint path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize vocabulary and model
    vocabulary = Vocabulary()
    vocabulary.load_vocab("path_to_your_vocab_file.txt")  # Replace with your vocabulary loading logic

    model = EncodertoDecoder(embed_size, hidden_size, len(vocabulary), num_layers)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # Set up transforms for inference
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.CenterCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Generate caption for the image
    caption = generate_caption(image_path, model, vocabulary, device, transform)
    print(f"Generated Caption: {caption}")

if __name__ == "__main__":
    main()