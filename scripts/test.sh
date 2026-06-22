if [ "$1" = "0" ]; then
    python3 a2part2.py --train --text_path x_train.txt --label_path y_train.txt --model_path model.pt
elif [ "$1" = "1" ]; then
    echo "Running Tests"
    echo "-------------------------"
    echo "Test 1. Domain: IMDB Reviews. Size: 25000 sequences."
    echo "Testing:"
    python3 a2part2.py --test --text_path x_test.txt --model_path model.pt --output_path out.txt
    python3 eval.py out.txt y_test.txt
fi
