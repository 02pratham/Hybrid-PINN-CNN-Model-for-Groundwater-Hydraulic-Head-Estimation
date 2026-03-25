# Visual Checker for Generated PiNN data
import h5py
import matplotlib.pyplot as plt
f = h5py.File('data/pinn_data/sample_00000089.h5','r')
fields = f['fields'][:]  # (4,nz,nx)
h_ref = f['h_ref'][:]
plt.figure(figsize=(12,6))
plt.subplot(1,5,1); plt.imshow(fields[0]); plt.title('c (kPa)'); plt.colorbar()
plt.subplot(1,5,2); plt.imshow(fields[1]); plt.title('phi (deg)'); plt.colorbar()
plt.subplot(1,5,3); plt.imshow(fields[2]); plt.title('gamma'); plt.colorbar()
plt.subplot(1,5,4); plt.imshow(fields[3]); plt.title('log10(k)'); plt.colorbar()
plt.subplot(1,5,5); plt.imshow(h_ref); plt.title('h_ref'); plt.colorbar()
plt.tight_layout(); plt.show()