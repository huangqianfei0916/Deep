import os
import io
import subprocess


class HadoopTool(object):
    """hadoop utils"""
    def __init__(self, hadoop_bin, fs_name, fs_ugi):
        self.hadoop_bin = hadoop_bin
        self.fs_name = fs_name
        self.fs_ugi = fs_ugi

    def ls(self, path):
        """ hdfs_ls """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -ls " + path
        cmd += " | awk '{print $8}'"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        stdout = io.TextIOWrapper(p.stdout, encoding='utf-8')
        filelist = stdout.read().split()
        return filelist

    def open(self, filename):
        """ cat """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -cat " + filename
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    def get(self, dst_file, src_file):
        """get"""
        cmd = self.hadoop_bin + " fs -Ddfs.client.block.write.retries=15"
        cmd += " -Ddfs.rpc.timeout=300000"
        cmd += " -Ddfs.delete.trash=1"
        cmd += " -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -get " + dst_file + " " + src_file + " 2>>hdfs.err "
        ret = os.system(cmd)
        return ret

    def upload(self, src_file, dst_file):
        """put"""
        cmd = self.hadoop_bin + " fs -Ddfs.client.block.write.retries=15"
        cmd += " -Ddfs.rpc.timeout=300000"
        cmd += " -Ddfs.delete.trash=1"
        cmd += " -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -put " + src_file + " " + dst_file + " 2>>hdfs.err "
        ret = os.system(cmd)
        return ret

    def rmr(self, dst_file):
        """rmr"""
        cmd = self.hadoop_bin + " fs -Ddfs.client.block.write.retries=15"
        cmd += " -Ddfs.rpc.timeout=300000"
        cmd += " -Ddfs.delete.trash=1"
        cmd += " -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -rmr " + dst_file
        ret = os.system(cmd)
        return ret