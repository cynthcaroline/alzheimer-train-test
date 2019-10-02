from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from finetuned_model import vgg16_finetuned

import argparse

def main(num_classes, batch_size, epochs):
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
                'data/mri/train',
                target_size=(150, 150),
                batch_size=batch_size,
                class_mode='categorical')
    
    validation_generator = datagen.flow_from_directory(
                'data/mri/validation',
                target_size=(150, 150),
                batch_size=batch_size,
                class_mode='categorical')

    model = vgg16_finetuned(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
    

    # callbacks
    base_path = 'data/'
    patience = 50
    log_file_path = base_path + 'logs/training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                    patience=int(patience/4), verbose=1)
    early_stop = EarlyStopping('val_loss', patience=patience)
    trained_models_path = base_path + 'pretrained/_vgg16_'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                        save_best_only=True)
    callbacks = [model_checkpoint, csv_logger,reduce_lr, early_stop]

    # Train model
    model.fit_generator(generator=train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch = 5122 // batch_size,
			validation_steps = 1282 // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks = callbacks
                        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Face Face Model')
    parser.add_argument('--numclasses', '-n', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=10000,
                        help='Number of sweeps over the dataset to train')
    args = parser.parse_args()

    main(args.numclasses,args.batchsize,args.epochs)

