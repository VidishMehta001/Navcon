$(window).scroll(function () {
var sc = $(window).scrollTop()
if (sc > 30) {
    $(".headerSection").addClass("fixed-header")
} else {
    $(".headerSection").removeClass("fixed-header")
    }
});

$('.navtggle').click(function(){
  $('body').toggleClass('actNav')
});



$(window).scroll(function() {
    if ($(this).scrollTop() > 50){
        $('.header-sec').addClass("sticky");
        $('.sub-links').css("top", "60px");
    }
    else{
        $('.header-sec').removeClass("sticky");
        $('.sub-links').css("top", "69px");
    }
});

var swiper = new Swiper('.spo-comp .swiper-container', {
  loop: true,
  navigation: {
    nextEl: '.swiper-button-next',
    prevEl: '.swiper-button-prev',
  },
  breakpoints: {
    768: {
      slidesPerView: 2,
      spaceBetween: 10,
    },
    992: {
      slidesPerView: 3,
    },
    1199: {
      slidesPerView: 4,
    },
  }
});

var swiper = new Swiper('.gallery-truck',{
  loop: true,
  navigation: {
    nextEl: '.swiper-button-next',
    prevEl: '.swiper-button-prev',
  },
  breakpoints: {
    480: {
      slidesPerView: 1,
    },
    768: {
      slidesPerView: 3,
    }
  }
});

var swiper = new Swiper('.news-feed .swiper-container',{
  loop: true,
  spaceBetween: 30,
  navigation: {
    nextEl: '.swiper-button-next',
    prevEl: '.swiper-button-prev',
  },
  breakpoints: {
    480: {
      slidesPerView: 1,
    },
    768: {
      slidesPerView: 3,
    }
  }
});

var swiper = new Swiper('.volta-swiper',{
  loop: true,
  navigation: {
    nextEl: '.swiper-button-next',
    prevEl: '.swiper-button-prev',
  },
  breakpoints: {
    480: {
      slidesPerView: 1,
    },
    768: {
      slidesPerView: 2,
    }
  }
});

// $(function(){
//
//     $('.at-drop-down').on('click', function(e){
//         if(Modernizr.mq('screen and (max-width:991px)')) {
//             $('.at-drop-down').parent('li').addClass('open');
//         }
//     })
//
// });


  if ($(window).width() > 991) {

  } else {
    $(".at-drop-down").click(function() {
      $('.sub-links').slideToggle();

    });
  }

  $("#spec_form_magellan").submit(function(e) {
    e.preventDefault();
  
    var $form = $(this);
    $.post($form.attr("action"), $form.serialize()).then(function() {
      // Download magellan specification form
      $("#mag-submit").removeClass("btn-primary");
      $("#mag-submit").addClass("disabled");
      $("#mag-submit").text("Please check your inbox for specifications.");
      setTimeout(() => {
        window.location.replace("/mag.html");
      }, 5);
    });
  });


  $("#spec_form_copernicus").submit(function(e) {
    e.preventDefault();
    var $form = $(this);
    $.post($form.attr("action"), $form.serialize()).then(function() {
      // Download copernicus specification form
      $("#copernicus-submit").removeClass("btn-primary");
      $("#copernicus-submit").addClass("disabled");
      $("#copernicus-submit").text("Please check your inbox for specifications.");
      setTimeout(() => {
        window.location.replace("/copernicus.html");
      }, 5);
    });
  });

  $("#spec_form_volta").submit(function(e) {
    e.preventDefault();
    var $form = $(this);

    $.post($form.attr("action"), $form.serialize()).then(function() {
      // Download volta specification form
      $("#volta-submit").removeClass("btn-primary");
      $("#volta-submit").addClass("disabled");
      $("#volta-submit").text("Please check your inbox for specifications.");
      setTimeout(() => {
        window.location.replace("/volta.html");
      }, 5);
    });
  });


  $("#contact_form").submit(function(e) {
    e.preventDefault();
  
    var $form = $(this);
    $.post($form.attr("action"), $form.serialize()).then(function() {
      // nothing
      window.location.replace("/contact.html");
    });
  });