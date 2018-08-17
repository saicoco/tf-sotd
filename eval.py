import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms
from skimage.morphology import label as bwlabel

tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

# sotd_generate_map
def relax_wrt_border( raw_bbox, height, width, border_perc = .16 ) :
    # load bbox
    top, bot, left, right = raw_bbox
    # compute box width and height
    box_w, box_h = right - left + 1, bot - top + 1
    # compute border width
    d = np.ceil( min( box_w, box_h ) * ( 1./ ( 1 - border_perc * 2 ) - 1 ) * 0.5 )
    # relax according to border info
    left, right  = max( 0, left - d ), min( right + d, width )
    top,  bot    = max( 0, top - d ), min( bot + d, height )
    return [ top, bot, left, right ]

def from_res_map_to_bbox( res_map, th_size = 8, th_prob = 0.5, border_perc = .16 ) :
    height, width = res_map.shape[:2]
    labels = res_map.argmax( axis = -1 )
    text = labels == 2
    bwtext, nb_regs = bwlabel( text, return_num = True )
    lut = { 'bounding_box' : [], 'proba' : [] }
    for reg_id in range( 1, nb_regs + 1 ) :
        row_idx, col_idx = np.nonzero( bwtext == reg_id )
        # get four corners
        left, right = col_idx.min(), col_idx.max() + 1
        top, bot = row_idx.min(), row_idx.max() + 1
        # relax w.r.t. border
        bbox = relax_wrt_border( [ top, bot, left, right ], height, width, border_perc )
        by0, by1, bx0, bx1 = bbox
        bh, bw = by1 - by0 + 1, bx1 - bx0 + 1
        # estimate text proba
        proba = np.median( res_map[top:bot, left:right, 2 ] )
        if ( proba >= th_prob ) and ( min( bh, bw ) >= th_size ):
            lut['bounding_box'].append( [ top, bot, left, right ] )
            lut['proba'].append( float( proba ) )
    return lut


def generate_boxes_from_map(sotd_map):
    print("sotd-map_shape:", sotd_map.shape)
    bg_map, border_map, center_map = sotd_map[0, :, :, 0], sotd_map[0, :, :, 1], sotd_map[0, :, :, 2]
    text_area = border_map + center_map
    text_area[text_area>0] = 1
    print("text_area_shape:", text_area.shape)

    # text_area = np.bitwise_or(center_map, border_map)
    image, contours, hierarchy = cv2.findContours(text_area.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rotrect = cv2.minAreaRect(contours[0])
    boxes = cv2.boxPoints(rotrect)
    boxes = np.int0(boxes)

    return boxes


def main(argv=None):
    import os
    print '!!!!', FLAGS.gpu_list
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_sotd = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            print "!!!!", ckpt_state
            print "!!!!", FLAGS.checkpoint_path
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()

                sotd_map = sess.run([f_sotd], feed_dict={input_images: [im_resized]})
                sotd_img = np.array(sotd_map[0][0,:,:,:]*255).astype(np.uint8)
                sotd_img = cv2.resize(sotd_img, dsize=(im_resized.shape[1], im_resized.shape[0]))
                cv2.imwrite(FLAGS.output_dir + '/'+os.path.basename(im_fn).split('.')[0]+'.png', sotd_img)
                boxes = generate_boxes_from_map(sotd_map[0])
                print("lenth of sotd_boxes", len(boxes))
                #print score
                #print geometry

                if boxes is not None:
                    print('length_boxes:', len(boxes))
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))

                # save to file
                if boxes is not None:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        '{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))

                    with open(res_file, 'w') as f:
                        for box in boxes:
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))
                            #print box
                            #print [box.astype(np.int32).reshape((-1, 1, 2))]
                            #print '******************************'
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])

def test_generate_box_from_map(img, sotd_map):
    boxes = generate_boxes_from_map(sotd_map)
    print "generate box"
    for box in boxes:
        box = sort_poly(box.astype(np.int32))
        cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    cv2.imwrite('tmp_res.jpg', img[:, :, ::-1])
if __name__ == '__main__':
    tf.app.run()
    '''
    import sys
    img = cv2.imread(str(sys.argv[1]))
    sotd_map = cv2.imread(str(sys.argv[2]))
    print 'sotd_map_shape:', sotd_map.shape
    test_generate_box_from_map(img, sotd_map)
    '''
