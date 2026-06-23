if [ "$1" = "0" ]; then
    python3 mlp_inference.py --train --text_path ../data/x_train.txt --label_path ../data/y_train.txt --model_path model.pt
elif [ "$1" = "1" ]; then
    echo "Running Tests"
    echo "-------------------------"
    echo "Test 1. Domain: IMDB Reviews. Size: 25000 sequences."
    echo "Testing:"
    python3 mlp_inference.py --test --text_path ../data/x_test.txt --model_path model.pt --output_path out.txt
    python3 mlp_eval.py out.txt ../data/y_test.txt
fi
