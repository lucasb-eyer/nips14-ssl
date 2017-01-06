from sys import argv, exit, stdout
import argparse

import os, time, numpy as np, h5py

# NOTE: some imports are done inside the code.
# For one, to avoid dependencies if unnecessary (don't need PIL if not saving images)
# and for another to only import theano after argument parsing.

parser = argparse.ArgumentParser(description='Generate a video dataset.')
parser.add_argument('dataset', choices=['mnist', 'svhn'], default='mnist',
                    help='the dataset to be generated.')
parser.add_argument('outfile',
                    help='Where to store the dataset numpy file.')
parser.add_argument('-n', '--nstreams', type=int, default=100,
                    help='Number of video streams to generate.')
parser.add_argument('-t', '--duration', type=int, default=600,
                    help='Duration (length) of each stream, in frames.')
parser.add_argument('-p', '--period', type=float, default=50,
                    help='Time until peaking at the next class, in frames.')
parser.add_argument('-s', '--period-std', type=float, default=0,
                    help='Standard deviation of the period for randomization.')
parser.add_argument('-c', '--classes', choices=['fixed', 'cycle', 'random'], default='cycle',
                    help='How to move through the classes.')
parser.add_argument('-i', '--interpolation', choices=['linear', 'harmonic'], default='linear',
                    help='How to interpolate between the classes.')
parser.add_argument('-l', '--latent', choices=['fixed', 'fly'], default='fixed',
                    help='How to move through latent z-space.')
parser.add_argument('--zvar', type=lambda x: np.fromstring(x, sep=' ')[:,None], default=np.array([[0.06]]),
                    help='Variance of z flight. Default 0.06 from paper.')
parser.add_argument('--zsmoothing', type=lambda x: np.fromstring(x, sep=' ')[:,None], default=np.array([[0.1]]),
                    help='Smoothing of z flight. Default 0.1 from paper.')
parser.add_argument('--images', action='store_true',
                    help='Also dump all images to the same folder as OUTFILE.')

args = parser.parse_args()
print(args)


def rowcols(n):
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))
    return nrows, ncols
tile_shape = rowcols(args.nstreams)


if args.dataset == 'svhn':
    n_x = 3*32*32
    dim_input = (32,32)
    n_y = 10

    colorImg = True
    binarize = False

    if False:
        n_hidden = (500,500)
        n_z = 300
        dir = 'models/svhn_yz_x_300-500-500/'
    else:
        n_hidden = (1000,1000)
        n_z = 300
        dir = 'models/svhn_yz_x_300-1000-1000/'

    from anglepy.models import GPUVAE_YZ_X
    import anglepy.ndict as ndict
    model = GPUVAE_YZ_X(None, n_x, n_y, n_hidden, n_z, n_hidden[::-1], 'softplus', 'softplus',
                        type_px='gaussian', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=100, init_sd=1e-2)
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    # PCA
    pca = ndict.loadz(dir+'pca_params.ndict.tar.gz')
    def f_dec(x):
        result = pca['eigvec'].dot(x * np.sqrt(pca['eigval'])) * pca['x_sd'] + pca['x_center']
        result = np.maximum(0, np.minimum(1, result))
        return result

elif args.dataset == 'mnist':
    n_x = 28*28
    dim_input = (28,28)
    n_y = 10
    n_hidden = (500,500)
    n_z = 50

    colorImg = False
    binarize = False

    dir = 'models/mnist_yz_x_50-500-500/'
    from anglepy.models import GPUVAE_YZ_X
    import anglepy.ndict as ndict
    model = GPUVAE_YZ_X(None, n_x, n_y, n_hidden, n_z, n_hidden[::-1], 'softplus', 'softplus',
                        type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=100, init_sd=1e-2)
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    f_dec = lambda x: x


# Test model
print("Test model")


def interp_linear(t, t_prev, t_next):
    return (t - t_prev)/(t_next - t_prev)


def interp_harmonic(t, t_prev, t_next):
    return 0.5 + 0.5*np.cos((1-interp_linear(t, t_prev, t_next))*np.pi)


interp_map = {
    'linear': interp_linear,
    'harmonic': interp_harmonic,
}


def interpolate_softmax(t, t_prev, t_next, idx_prev, idx_next, K, mode=interp_linear, dtype=np.float32):
    # NOTE: it's the other way around as the rest of this code, because I took
    #       it out of my own utils which follow a different convention.
    p_next = mode(t, t_prev, t_next)
    out = np.zeros((len(idx_prev), K), dtype)
    # NOTE: adding in case prev == next
    out[np.arange(len(idx_prev)), idx_prev] += 1 - p_next
    out[np.arange(len(idx_prev)), idx_next] += p_next
    return out


class ClassFixed:
    def __init__(self, n_streams, n_y):
        self.py = np.zeros((n_streams, n_y), np.float32)
        self.py[np.arange(n_streams), np.random.randint(0, n_y, n_streams)] = 1

    def next(self):
        pass

    def get_softmax(self):
        return np.array(self.py.T)


class ClassCycler:
    def __init__(self, n_streams, n_y, period, interp_fn):
        self.n_y = n_y
        self.interp_fn = interp_fn
        self.yidx_prev = np.random.randint(n_y, size=n_streams)
        self._comp_next()
        self.t_prev = 0
        self.t_next = period - 1
        self.t = 0

    def _comp_next(self):
        self.yidx_next = (self.yidx_prev + 1) % self.n_y

    def next(self):
        self.t += 1
        if self.t_next < self.t:
            self.t = 0
            self.yidx_prev = self.yidx_next
            self._comp_next()

    def get_softmax(self):
        return interpolate_softmax(self.t, self.t_prev, self.t_next,
                                   self.yidx_prev, self.yidx_next, self.n_y,
                                   self.interp_fn).T  # .T: See comment inside


class ClassRandom:
    def __init__(self, n_streams, n_y, period_mean, period_std, interp_fn):
        self.n_y = n_y
        self.interp_fn = interp_fn
        self.period_mean, self.period_std = period_mean, period_std
        self.yidx_prev = np.random.randint(0, self.n_y, size=n_streams)
        self.yidx_next = np.random.randint(0, self.n_y, size=n_streams)
        self.t_prev = np.zeros(n_streams)
        self.t_next = self._next_t(n_streams)
        self.t = 0

    def _next_t(self, n):
        return np.maximum(2, np.round(self.period_mean + self.period_std*np.random.randn(n)))

    def next(self):
        self.t += 1
        for i in np.where(self.t > self.t_next)[0]:
            self.t_prev[i] = self.t_next[i]
            self.t_next[i] += self._next_t(1)
            self.yidx_prev[i] = self.yidx_next[i]
            self.yidx_next[i] = np.random.randint(self.n_y)

    def get_softmax(self):
        return interpolate_softmax(self.t, self.t_prev, self.t_next,
                                   self.yidx_prev, self.yidx_next, self.n_y,
                                   self.interp_fn).T  # .T: See comment inside


if args.classes == 'fixed':
    cls = ClassFixed(args.nstreams, n_y)
elif args.classes == 'cycle':
    cls = ClassCycler(args.nstreams, n_y, args.period, interp_map[args.interpolation])
elif args.classes == 'random':
    cls = ClassRandom(args.nstreams, n_y, args.period, args.period_std, interp_map[args.interpolation])
else:
    assert False


class LatentFixed:
    def __init__(self, n_streams, n_z):
        z = np.random.standard_normal((n_z, n_streams))
        z = np.sqrt(1-args.zvar)*z + np.sqrt(args.zvar)*np.random.randn(*z.shape)
        self.z = z.astype(np.float32)

    def next(self):
        pass

    def get_z(self):
        return self.z


class LatentRandom:
    def __init__(self, n_streams, n_z, zvar, zsmoothing):
        self.z = np.random.randn(n_z, n_streams)
        self.zsmooth = self.z.copy()
        self.smoothingfactor = zsmoothing
        self.zvar = zvar

    def next(self):
        # Do step of Gaussian diffusion process
        self.z = np.sqrt(1 - self.zvar)*self.z + np.sqrt(self.zvar)*np.random.randn(*self.z.shape)
        # Smooth the trajectory
        self.zsmooth += self.smoothingfactor*(self.z - self.zsmooth)

    def get_z(self):
        return self.zsmooth.astype(np.float32)


if args.latent == 'fixed':
    latent = LatentFixed(args.nstreams, n_z)
elif args.latent == 'fly':
    latent = LatentRandom(args.nstreams, n_z, args.zvar, args.zsmoothing)
else:
    assert False

outdir = os.path.dirname(os.path.abspath(args.outfile))
os.makedirs(outdir, exist_ok=True)

# Not using compression as it makes this much slower. If we want compression, use compression='lzf'.
# lzf is much faster than gzip and compresses only a little worse. Compression factor on mnist: ~50%
h5_f = h5py.File(os.path.abspath(args.outfile), "w")
h5_x = h5_f.create_dataset("x", (args.duration, args.nstreams, n_x), dtype=np.float32, fletcher32=True)
h5_y = h5_f.create_dataset("y", (args.duration, args.nstreams, n_y), dtype=np.float32, fletcher32=True)
h5_z = h5_f.create_dataset("z", (args.duration, args.nstreams, n_z), dtype=np.float32, fletcher32=True)

# Store all commandline arguments for posteriority.
for k, v in vars(args).items():
    h5_f.attrs[k] = v
h5_f.attrs['t'] = 0

for t in range(args.duration):
    y = cls.get_softmax()
    z = latent.get_z()

    # Run the model (only the decoder part).
    _, _, _z_confab = model.gen_xz({'y': y}, {'z': z}, n_batch=args.nstreams)
    x_samples = f_dec(_z_confab['x'])

    h5_x[t] = x_samples.T
    h5_y[t] = y.T
    h5_z[t] = z.T

    # Keep track of progress.
    h5_f.attrs['t'] = t

    stdout.write('\r{}/{}'.format(t+1, args.duration)) ; stdout.flush()

    # Save individual images and imagegrids.
    if args.images:
        # Each individual image first.
        import PIL.Image
        for istream, x in enumerate(x_samples.T):
            if colorImg:
                x = (x.reshape((3,) + dim_input)*255).astype(np.uint8).transpose([1, 2, 0])
                ximg = PIL.Image.fromarray(x, mode='RGB')
            else:
                x = (x.reshape(dim_input)*255).astype(np.uint8)
                ximg = PIL.Image.fromarray(x, mode='L')
            ximg.save('{}/{}-{}.png'.format(outdir, istream, t), 'PNG')

        # Then, the whole image-grid (used to make a video later on).
        from anglepy.paramgraphics import mat_to_img
        image = mat_to_img(x_samples, dim_input, colorImg=colorImg, tile_shape=tile_shape)
        # Make sure the nr of rows and cols are even
        width, height = image.size
        if width%2==1: width += 1
        if height%2==1: height += 1
        image = image.resize((width, height))
        # Save it
        image.save("{}/{}.png".format(outdir, t), 'PNG')

    cls.next()
    latent.next()

print()

# Make video
if args.images:
    os.system("avconv -i {outdir}/%d.png -c:v libx264 -pix_fmt yuv420p -r 30 '{outdir}/video.mp4'".format(outdir=outdir))
print("Saved to {}".format(outdir))
print("Done.")

