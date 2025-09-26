pkg load image   % só precisa rodar se o pacote não estiver carregado

imagefiles = dir('*.png'); % veja o tipo das imagens e altere, neste caso temos o .png
[num, z] = size(imagefiles) ; % num vai ter o total de imagens...


for i = 1: num
  image = imread( imagefiles(i).name) ; % carrega a imagem...
end
