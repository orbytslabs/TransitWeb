$(document).ready(function (e) {
    $('#finderImage').click(function (e) {
        var posX = $(this).offset().left,
            posY = $(this).offset().top;
        var sourcePosX = e.pageX - posX,
            sourcePosY = e.pageY - posY;
        alert(sourcePosX + ' , ' + sourcePosY);

        $.ajax({
          type: "POST",
          url: "/source-selection/create-photometry-source",
          data: { 
            "sourcePosX": sourcePosX, 
            "sourcePosY": sourcePosY,
            csrfmiddlewaretoken: $("input[name='csrfmiddlewaretoken']").val() 
          },
          success: function(response){
            console.log('Success');
          },
        });
    });
});