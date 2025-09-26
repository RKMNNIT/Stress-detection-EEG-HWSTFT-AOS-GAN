import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from sklearn.metrics import accuracy_score, classification_report

def AOS_GAN(epochs=200, batch_size=64, latent_dim=100):
    mat = scipy.io.loadmat('Selected_Features.mat')
    X = mat['selected_features']
    y = mat['labels'].ravel()

    X = (X - np.min(X)) / (np.max(X) - np.min(X))

    def build_generator():
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=latent_dim),
            layers.Dense(X.shape[1], activation='tanh')
        ])
        return model

    def build_discriminator():
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=X.shape[1]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(learning_rate=1e-4),
                      metrics=['accuracy'])
        return model

    generator = build_generator()
    discriminator = build_discriminator()

    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    generated = generator(gan_input)
    gan_output = discriminator(generated)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=1e-4))

    for epoch in range(epochs):
        idx = np.random.randint(0, X.shape[0], batch_size)
        real_samples, real_labels = X[idx], y[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, D_loss: {d_loss_real[0]+d_loss_fake[0]:.4f}, G_loss: {g_loss:.4f}")

    y_pred = (discriminator.predict(X) > 0.5).astype(int).ravel()
    scipy.io.savemat('Predictions.mat', {'y_pred': y_pred, 'y_true': y})

    acc = accuracy_score(y, y_pred)
    print("\nClassification Report:\n", classification_report(y, y_pred))
    print(f"Final AOS-GAN Accuracy: {acc*100:.2f}%")
