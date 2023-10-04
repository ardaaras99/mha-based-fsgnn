# TODO

- [x] bu seedleri bir sayı seçip onu kullanmak lazım hep, sonradan aynı parametrelerle aynı resultı almamız lazım

- [x] gpuya bir şey koymadım macte yazıyordum, zaten aşırı basit yapması modeli ve matrixleri gpuya koycan o kadar

- [x] şuan wandb de parametreler nested olduğu için fazladan loglama problemi var, bazı columnlar boş kalıyor, bak bakalım nasıl çözersin

- [ ] Calculate mask matrix list combinations up to desired power, currently we calculate ``mask_matrix_list`` in each run, since some of the hyperparameter combinations will have same `max_hop` parameter, there is no need to calculate them for each run. Instead calculate all the combinations upto maximum number of value in `max_hop` (currently 8), then read from the disk.
