document.addEventListener('DOMContentLoaded', (event) => {
    var dragSrcEl = null;
    function handleDragStart(e) {
    // this.style.opacity = '0.4';
    
    dragSrcEl = this;

    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', this.innerHTML);
    }

    function handleDragOver(e) {
    if (e.preventDefault) {
        e.preventDefault();
    }

    e.dataTransfer.dropEffect = 'move';
    
    return false;
    }

    function handleDragEnter(e) {
    this.classList.add('over');
    }

    function handleDragLeave(e) {
    this.classList.remove('over');
    }

    function handleDrop(e) {
    if (e.stopPropagation) {
        e.stopPropagation(); // stops the browser from redirecting.
    }
    
    if (dragSrcEl != this) {
        dragSrcEl.innerHTML = this.innerHTML;
        this.innerHTML = e.dataTransfer.getData('text/html');
    }
    
    return false;
    }

    function handleDragEnd(e) {
    this.style.opacity = '1';
    
    
    items.forEach(function (item) {
        item.classList.remove('over');
    });
    }

    let items = document.querySelectorAll('.container .box');
    items.forEach(function(item) {
        
        item.addEventListener('dragstart', handleDragStart, false);
        item.addEventListener('dragenter', handleDragEnter, false);
        item.addEventListener('dragover', handleDragOver, false);
        item.addEventListener('dragleave', handleDragLeave, false);
        item.addEventListener('drop', handleDrop, false);
        item.addEventListener('dragend', handleDragEnd, false);
        console.log(item.firstElementChild.id)

    });
    console.log(fruits)
    });
    const fruits = [];

    // const fs = require('fs');
    function Done(){
        const fruits = [];
        let items = document.querySelectorAll('.container .box');
        
        console.log(items)
        items.forEach(function(item) {
            fruits.push(item.firstElementChild.id);
        });
        console.log(fruits)
        // fs.writeFile('output.txt', id, (err) => {
        var fs = require('fs');

        var file = fs.createWriteStream('array.txt');
        file.on('error', function(err) { /* error handling */ });
        fruits.forEach(function(v) { file.write(v.join(', ') + '\n'); });
        file.end();
        // In case of a error throw err.
        // if (err) throw err;
    // });
    }  