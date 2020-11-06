# -*- coding:utf-8 -*-
from absl import flags, app
from random import shuffle, random
from model import *

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import os

Add = tf.keras.layers.Add()
Mul = tf.keras.layers.Multiply()

flags.DEFINE_string("A_img_path", "/yuhwan/yuhwan/Dataset/AFAD_dataset/", "A input image path")

flags.DEFINE_string("A_txt_path", "/yuhwan/yuhwan/Dataset/[1]Third_dataset/[2]Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/female_16_39_train.txt", "A input text path")

flags.DEFINE_string("B_img_path", "/yuhwan/yuhwan/Dataset/[1]Third_dataset/Morph/All/male_40_63/", "B input image path")

flags.DEFINE_string("B_txt_path", "/yuhwan/yuhwan/Dataset/[1]Third_dataset/[2]Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/male_40_63_train.txt", "B input text path")

flags.DEFINE_integer("img_size", 256, "Input image size")

flags.DEFINE_integer("load_size", 266, "Input load size before cropping")

flags.DEFINE_integer("batch_size", 1, "Training batch size")

flags.DEFINE_integer("epochs", 200, "Number of training epochs")

flags.DEFINE_integer("epoch_decay", 100, "")

flags.DEFINE_float("lr", 0.0002, "Learning rate")

flags.DEFINE_float("L1_lambda", 10.0, "")

flags.DEFINE_bool("pre_checkpoint", True, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "checkpoint/Style_transform/3rd_paper/related_work/UACycleGAN/first_fold/AFAD-F_Morph-M_16_39_40_63/checkpoint/1263", "Pre checkpoint path for testing or continue the train")

flags.DEFINE_bool("train", False, "True or False")

flags.DEFINE_string("save_images", "/yuhwan/yuhwan/checkpoint/Style_transform/3rd_paper/related_work/UACycleGAN/first_fold/AFAD-F_Morph-M_16_39_40_63/sample_images/", "Saved checkpoint path")

flags.DEFINE_string("save_checkpoint", "/yuhwan/yuhwan/checkpoint/Style_transform/3rd_paper/related_work/UACycleGAN/first_fold/AFAD-F_Morph-M_16_39_40_63/checkpoint", "Save checkpoint path")
#############################################################################################################################################################################################
flags.DEFINE_string("A_test_txt", "/yuhwan/yuhwan/Dataset/[1]Third_dataset/[2]Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/test/female_16_39_test.txt", "A test text path")

flags.DEFINE_string("A_test_img", "/yuhwan/yuhwan/Dataset/AFAD_dataset/", "A test image path")

flags.DEFINE_string("A_output_img", "checkpoint/Style_transform/3rd_paper/related_work/UACycleGAN/first_fold/AFAD-F_Morph-M_16_39_40_63/test_images/AFAD-M_40_63", "A generated images")

flags.DEFINE_string("B_test_txt", "/yuhwan/yuhwan/Dataset/[1]Third_dataset/[2]Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/test/male_40_63_test.txt", "B test text path")

flags.DEFINE_string("B_test_img", "/yuhwan/yuhwan/Dataset/[1]Third_dataset/Morph/All/male_40_63/", "B test image path")

flags.DEFINE_string("B_output_img", "checkpoint/Style_transform/3rd_paper/related_work/UACycleGAN/first_fold/AFAD-F_Morph-M_16_39_40_63/test_images/Morph-F_16_39", "A generated images")

flags.DEFINE_string("dir", "A2B", "A2B or B2A")
#############################################################################################################################################################################################

FLAGS = flags.FLAGS
FLAGS(sys.argv)

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

len_dataset = len(np.loadtxt(FLAGS.A_txt_path, dtype=np.str, skiprows=0, usecols=1))
G_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.epoch_decay * len_dataset)
D_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.epoch_decay * len_dataset)

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
att_optim = tf.keras.optimizers.Adam(FLAGS.lr*0.1, beta_1=0.5)

def ThresHold(x, t=0.1):        # 우선 이게 의심됨
    mask = tf.cast(tf.greater_equal(x, t), tf.float32)
    return mask

def train_func(A, B):

    A_img = tf.io.read_file(A)
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3], seed=1234) / 127.5 - 1

    B_img = tf.io.read_file(B)
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.load_size, FLAGS.load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.img_size, FLAGS.img_size, 3], seed=1234) / 127.5 - 1

    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img)
        B_img = tf.image.flip_left_right(B_img)

    return A_img, B_img

def abs_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))

def mae_criterion_for_generator(input):
    return tf.reduce_mean(tf.math.squared_difference(input, 1))

def mae_criterion_for_discriminator(real, fake):
    return (tf.reduce_mean(tf.math.squared_difference(real, 1)) + tf.reduce_mean(tf.math.squared_difference(fake, 0))) * 0.5

#@tf.function
def run_model(model, images, training):
    output = model(images, training=training)
    return output

def train_step_(g_A_model, g_B_model, d_A_model, d_B_model, g_A_att_model, g_B_att_model, A_images, B_images, ep):

    with tf.GradientTape(persistent=True) as tape, tf.GradientTape(persistent=True) as d_tape:

        if ep <= 30:
            mask_A = run_model(g_A_att_model, A_images, True)
            mask_B = run_model(g_B_att_model, B_images, True)
        else:
            mask_A = run_model(g_A_att_model, A_images, False)
            mask_B = run_model(g_B_att_model, B_images, False)            
 
        mask_A = tf.concat([mask_A] * 3, 3)
        mask_B = tf.concat([mask_B] * 3, 3)

        mask_A_on_A = tf.math.multiply(A_images, ThresHold(mask_A))
        mask_B_on_B = tf.math.multiply(B_images, ThresHold(mask_B))

        prob_real_A_is_real = run_model(d_A_model, mask_A_on_A, True)
        prob_real_B_is_real = run_model(d_B_model, mask_B_on_B, True)
 
        fake_B_from_g = run_model(g_A_model, A_images, True)
        fake_B = tf.math.add(tf.math.multiply(fake_B_from_g, mask_A), tf.math.multiply(A_images, 1 - mask_A))

        fake_A_from_g = run_model(g_B_model, B_images, True)
        fake_A = tf.math.add(tf.math.multiply(fake_A_from_g, mask_B), tf.math.multiply(B_images, 1 - mask_B)) 

        mask_fakeA_on_B = tf.math.multiply(fake_A, ThresHold(mask_B))
        mask_fakeB_on_A = tf.math.multiply(fake_B, ThresHold(mask_A))

        prob_fake_A_is_real = run_model(d_A_model, mask_fakeA_on_B, True)
        prob_fake_B_is_real = run_model(d_B_model, mask_fakeB_on_A, True)

        if ep <= 30:
            mask_ACycle = run_model(g_A_att_model, fake_A, True)
            mask_BCycle = run_model(g_B_att_model, fake_B, True)
        else:
            mask_ACycle = run_model(g_A_att_model, fake_A, False)
            mask_BCycle = run_model(g_B_att_model, fake_B, False)

        mask_ACycle = tf.concat([mask_ACycle] * 3, 3)
        mask_BCycle = tf.concat([mask_BCycle] * 3, 3)
 
        mask_acycle_on_fakeA = tf.math.multiply(fake_A, ThresHold(mask_ACycle))
        mask_bcycle_on_fakeB = tf.math.multiply(fake_B, ThresHold(mask_BCycle))
 
        cycle_A_from_g = run_model(g_B_model, fake_B, True)
        cycle_B_from_g = run_model(g_A_model, fake_A, True)

        fake_A_ = tf.math.add(tf.math.multiply(cycle_A_from_g, mask_BCycle), tf.math.multiply(fake_B, 1 - mask_BCycle))
        fake_B_ = tf.math.add(tf.math.multiply(cycle_B_from_g, mask_ACycle), tf.math.multiply(fake_A, 1 - mask_ACycle))
        ################################################################        
 
        Generator_GAN_A_loss = mae_criterion_for_generator(prob_fake_A_is_real)
        Generator_GAN_B_loss = mae_criterion_for_generator(prob_fake_B_is_real)

        Cycle_A_loss = abs_criterion(A_images, fake_A_) * FLAGS.L1_lambda
        Cycle_B_loss = abs_criterion(B_images, fake_B_) * FLAGS.L1_lambda
    
        a2b_g_loss = Generator_GAN_B_loss + Cycle_A_loss + Cycle_B_loss
        b2a_g_loss = Generator_GAN_A_loss + Cycle_A_loss + Cycle_B_loss

        g_loss = Generator_GAN_A_loss + Generator_GAN_B_loss + Cycle_A_loss + Cycle_B_loss

        Discriminator_A_loss = mae_criterion_for_discriminator(prob_real_A_is_real, prob_fake_A_is_real)
        Discriminator_B_loss = mae_criterion_for_discriminator(prob_real_B_is_real, prob_fake_B_is_real)

        d_loss = Discriminator_A_loss + Discriminator_B_loss

    g_grad = tape.gradient(g_loss, g_A_model.trainable_variables + g_B_model.trainable_variables)
    d_grad = d_tape.gradient(d_loss, d_A_model.trainable_variables + d_B_model.trainable_variables)

    if ep <= 30:
        att_grad = tape.gradient(g_loss, g_A_att_model.trainable_variables + g_B_att_model.trainable_variables)

    g_optim.apply_gradients(zip(g_grad, g_A_model.trainable_variables + g_B_model.trainable_variables))
    d_optim.apply_gradients(zip(d_grad, d_A_model.trainable_variables + d_B_model.trainable_variables))

    if ep <= 30:
        att_optim.apply_gradients(zip(att_grad, g_A_att_model.trainable_variables + g_B_att_model.trainable_variables))

    return g_loss, d_loss

def main(argv=None):
    
    g_A_model = ResnetGenerator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    g_B_model = ResnetGenerator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    d_A_model = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    d_B_model = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    g_A_att_model = attention_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    g_B_att_model = attention_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    g_A_model.summary()
    d_A_model.summary()
    g_A_att_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(g_A_model=g_A_model,
                                   g_B_model=g_B_model,
                                   d_A_model=d_A_model,
                                   d_B_model=d_B_model,
                                   g_A_att_model=g_A_att_model,
                                   g_B_att_model=g_B_att_model,
                                   g_optim=g_optim,
                                   d_optim=d_optim,
                                   att_optim=att_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint!!!")

    if FLAGS.train:
        # input data
        A_img = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_img = [FLAGS.A_img_path + img for img in A_img]

        B_img = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_img = [FLAGS.B_img_path + img for img in B_img]

        count = 0
        for epoch in range(FLAGS.epochs):
            
            np.random.shuffle(A_img)
            np.random.shuffle(B_img)

            gener = tf.data.Dataset.from_tensor_slices((A_img, B_img))
            gener = gener.shuffle(len(B_img))
            gener = gener.map(train_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_iter = iter(gener)
            train_idx = len(A_img) // FLAGS.batch_size
            for step in range(train_idx):

                A_images, B_images = next(train_iter)

                g_loss, d_loss = train_step_(g_A_model, g_B_model, d_A_model, d_B_model, g_A_att_model, g_B_att_model, A_images, B_images, epoch)

                # if count % 10 == 0:
                print("Epoch: {} [{}/{}] g_loss = {}, d_loss = {}".format(epoch, step, train_idx, g_loss, d_loss))

                if count % 500 == 0:
                    mask_A = run_model(g_A_att_model, A_images, False)
                    mask_B = run_model(g_B_att_model, B_images, False)
                    mask_A = tf.concat([mask_A] * 3, 3)
                    mask_B = tf.concat([mask_B] * 3, 3)

                    mask_A_on_A = tf.multiply(A_images, mask_A)
                    mask_B_on_B = tf.multiply(B_images, mask_B)

                    fake_B_from_g = run_model(g_A_model, A_images, False)
                    fake_B = tf.multiply(fake_B_from_g, mask_A) + tf.multiply(A_images, 1 - mask_A)

                    fake_A_from_g = run_model(g_B_model, B_images, False)
                    fake_A = tf.multiply(fake_A_from_g, mask_B) + tf.multiply(B_images, 1 - mask_B)

                    plt.imsave(FLAGS.save_images + "{}_fake_A.jpg".format(count), fake_A[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.save_images + "{}_fake_B.jpg".format(count), fake_B[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.save_images + "{}_input_A.jpg".format(count), A_images[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.save_images + "{}_input_B.jpg".format(count), B_images[0].numpy() * 0.5 + 0.5)

                if count % 1000 == 0:
                    number = int(count/1000)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, number)
                    ckpt = tf.train.Checkpoint(g_A_model=g_A_model,
                                                g_B_model=g_B_model,
                                                d_A_model=d_A_model,
                                                d_B_model=d_B_model,
                                                g_A_att_model=g_A_att_model,
                                                g_B_att_model=g_B_att_model,
                                                g_optim=g_optim,
                                                d_optim=d_optim,
                                                att_optim=att_optim)
                    if not os.path.isdir(model_dir):
                        print("Make {} files to save the checkpoint".format(number))
                        os.makedirs(model_dir)
                    ckpt_dir = model_dir + "/" + "UA_CycleGAN_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)


                count += 1
    else:
        A_data = np.loadtxt(FLAGS.A_test_txt, dtype="<U200", skiprows=0, usecols=0)
        A_name = A_data
        A_data = [FLAGS.A_test_img + img for img in A_data]

        B_data = np.loadtxt(FLAGS.B_test_txt, dtype="<U200", skiprows=0, usecols=0)
        B_name = B_data
        B_data = [FLAGS.B_test_img + img for img in B_data]

        if FLAGS.dir is "A2B":
            print("====================")
            print("Direction --> A to B")
            print("====================")

            data = tf.data.Dataset.from_tensor_slices(A_data)
            data = data.map(test_img)
            data = data.batch(1)
            data = data.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(data)
            for i in range(len(A_data)):
                img = next(it)

                mask_A = run_model(g_A_att_model, img, False)
                mask_A = tf.concat([mask_A] * 3, 3)

                fake_B_from_g = run_model(g_A_model, img, False)
                fake_B = tf.multiply(fake_B_from_g, mask_A) + tf.multiply(img, 1 - mask_A)

                plt.imsave(FLAGS.A_output_img + "/" + A_name[i], fake_B[0].numpy() * 0.5 + 0.5)

                if i % 1000 == 0:
                    print("(A2B)Generated {} images...".format(i))

        else:
            print("====================")
            print("Direction --> B to A")
            print("====================")

            data = tf.data.Dataset.from_tensor_slices(B_data)
            data = data.map(test_img)
            data = data.batch(1)
            data = data.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(data)
            for i in range(len(B_data)):
                img = next(it)

                mask_B = run_model(g_B_att_model, img, False)
                mask_B = tf.concat([mask_B] * 3, 3)

                fake_A_from_g = run_model(g_B_model, img, False)
                fake_A = tf.multiply(fake_A_from_g, mask_B) + tf.multiply(img, 1 - mask_B)

                plt.imsave(FLAGS.B_output_img + "/" + B_name[i], fake_A[0].numpy() * 0.5 + 0.5)

                if i % 1000 == 0:
                    print("(B2A)Generated {} images...".format(i))

def test_img(data_list):
    img = tf.io.read_file(data_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.convert_image_dtype(img, tf.float32) / 127.5 - 1.

    return img

if __name__ == "__main__":
    app.run(main)
