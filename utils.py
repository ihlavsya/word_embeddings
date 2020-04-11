import os
from itertools import product

from DataGenerator import DataGenerator


def get_filenames(directory_path):
    filenames = []
    for root, dirs, files in os.walk(directory_path):
        for name in files:
            if not name.startswith('.'):
                filenames.append(os.path.join(root, name))
    return filenames


def write_into_one_big_file(out_filename, in_filenames):
    prefix = ('From:', 'Subject:', 'Organization:', 'Lines:', 'In article')
    with open(out_filename, 'w') as outfile:
        for name in in_filenames:
            with open(name, 'r') as infile:
                text = []
                try:
                    for line in infile:
                        if not line.startswith(prefix):
                            text.append(line)
                except UnicodeDecodeError as e:
                    print(e)
                    print(name)
                    continue
                # remove author
                outfile.writelines(text[:-3])


def prepare_data():
    dir_path = 'dataset/20news-bydate'
    filenames = get_filenames(dir_path)
    write_into_one_big_file('dataset/train.txt', filenames)


def plot_losses(losses):
    plt.plot(losses)
    plt.show()


def train_everything_and_save(text):
    embedding_dim = 10
    data_generator = DataGenerator(text, window_size=5, is_connection_map=True)
    #losses1 = train_global_n_gram(data_generator, embedding_dim)
    print('1st loss done')
    #losses2 = train_n_gram(data_generator, embedding_dim)
    print('2nd loss done')
    losses3 = train_cbow(data_generator, embedding_dim)
    print('3rd loss done')

    #np.savetxt('losses1.txt', np.array(losses1), delimiter=',')
    plot_losses(losses3)
    #np.savetxt('losses2.txt', np.array(losses2), delimiter=',')
    #np.savetxt('losses3.txt', np.array(losses3), delimiter=',')