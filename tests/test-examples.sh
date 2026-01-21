cd ../examples

for TEST in baregpt clevr hydra; do
	cd $TEST
	echo "Running $TEST..."
	if python ./run.py > /dev/null ; then
	echo "$TEST OK"
	else	echo "$TEST FAIL"
	fi

	cd ..
done

# mlp
if sheaf ./mlp/run.shf  > /dev/null ; then
	echo "mlp OK"
else
   	echo "mlp FAIL"
fi
