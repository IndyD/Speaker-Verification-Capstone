import pickle
import sys
import pdb
from sklearn.model_selection import train_test_split

import siamese_model
import utils




if __name__ == '__main__':
    PARAMS = utils.config_init(sys.argv)
    (pairs, labels) = utils.load(PARAMS.PATHS.PAIRS_PATH)
    pairs_train, pairs_test, label_train, label_test = train_test_split(
        pairs, labels, test_size=PARAMS.DATA_GENERATOR.TEST_SPLIT, random_state=1
    )

    IMG_SHAPE = (
        PARAMS.DATA_GENERATOR.N_MELS,
        PARAMS.DATA_GENERATOR.MAX_FRAMES,
        1
    )

    model = siamese_model.build_siamese_vgg7_model(IMG_SHAPE)

    pdb.set_trace()
    #model.compile(loss=siamese_model.contrastive_loss, optimizer="adam")
    model.compile(loss=siamese_model.contrastive_loss_with_margin(margin=1), optimizer="adam")
    print("Training model...")
    history = model.fit(
        [pairs_train[:, 0], pairs_train[:, 1]], label_train[:],
        validation_data=([pairs_test[:, 0], pairs_test[:, 1]], label_test[:]),
        batch_size=PARAMS.TRAINING.BATCH_SIZE,
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
        )
    pdb.set_trace()
