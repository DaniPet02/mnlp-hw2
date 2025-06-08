#ssh-keygen -f '/home/andrea/.ssh/known_hosts' -R 'login.leonardo.cineca.it'
eval $(ssh-agent)
step ssh login pizzi.1995517@studenti.uniroma1.it --provisioner cineca-hpc
ssh apizzi00@login.leonardo.cineca.it
